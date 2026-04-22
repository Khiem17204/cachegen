"""Shared TTFT benchmark core for the CLI and Colab notebook."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import torch

# Prefer vendored vLLM source if present.
SCRIPT_DIR = Path(__file__).resolve().parent
VENDORED_VLLM = SCRIPT_DIR / "third_party" / "vllm"
if VENDORED_VLLM.exists():
    sys.path.insert(0, str(VENDORED_VLLM))

_VLLM_IMPORT_ERROR: Exception | None = None
try:
    import vllm
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.config import KVTransferConfig
except Exception as exc:  # pragma: no cover - import environment dependent
    vllm = None
    AsyncEngineArgs = AsyncLLMEngine = SamplingParams = KVTransferConfig = None
    _VLLM_IMPORT_ERROR = exc

_TOKENIZER_IMPORT_ERROR: Exception | None = None
try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover - import environment dependent
    AutoTokenizer = None
    _TOKENIZER_IMPORT_ERROR = exc

PROMPT_CORPUS_PATH = SCRIPT_DIR / "benchmarks" / "prompt_corpus.txt"
DEFAULT_RESULTS_PATH = SCRIPT_DIR / "benchmarks" / "results.json"
DEFAULT_CACHE_ROOT = SCRIPT_DIR / "benchmarks" / "cache_store"
DEFAULT_CLAIM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CLAIM_TARGET_BANDWIDTH_MBPS = 3000.0
CLAIM_SPEEDUP_RANGE = (3.2, 3.7)
DEFAULT_PROMPT_INSTANCES = 2
CLAIM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ModelConfig:
    model: str
    max_model_len: int = 8192
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class TransferModeConfig:
    name: str
    cachegen_enabled: bool
    kv_cache_dtype: str
    uses_kv_transfer: bool = False
    paper_facing: bool = True


MODE_CONFIGS: dict[str, TransferModeConfig] = {
    "quantized_fp8": TransferModeConfig(
        name="quantized_fp8",
        cachegen_enabled=False,
        kv_cache_dtype="fp8",
        uses_kv_transfer=False,
    ),
    "cachegen": TransferModeConfig(
        name="cachegen",
        cachegen_enabled=True,
        kv_cache_dtype="auto",
        uses_kv_transfer=True,
    ),
    "raw_debug": TransferModeConfig(
        name="raw_debug",
        cachegen_enabled=False,
        kv_cache_dtype="auto",
        uses_kv_transfer=False,
        paper_facing=False,
    ),
}


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    length_tokens: int
    instance_index: int
    offset_tokens: int
    prompt_token_ids: list[int]


@dataclass(frozen=True)
class BenchmarkConfig:
    models: list[ModelConfig]
    modes: list[str]
    bandwidth_mbps: list[float]
    prompt_lengths: list[int]
    repeats: int = 5
    max_tokens: int = 16
    stop: list[str] = field(default_factory=list)
    tensor_parallel_size: int = 1
    quantization: str | None = None
    gpu_memory_utilization: float = 0.85
    cache_root: Path = DEFAULT_CACHE_ROOT
    output_path: Path = DEFAULT_RESULTS_PATH
    prompt_corpus_path: Path = PROMPT_CORPUS_PATH
    prompt_instances_per_length: int = DEFAULT_PROMPT_INSTANCES
    claim_model: str = DEFAULT_CLAIM_MODEL
    claim_bandwidth_mbps: float = CLAIM_TARGET_BANDWIDTH_MBPS


@dataclass
class RunResult:
    model: str
    mode: str
    repetition: int
    prompt_id: str
    prompt_length_tokens_requested: int
    prompt_length_tokens: int
    prompt_instance_index: int
    prompt_offset_tokens: int
    bandwidth_mbps: float
    output_tokens: int
    cached_tokens: int
    ttft_seconds: float
    end_to_end_seconds: float
    tokens_per_second: float
    raw_tensor_bytes: int
    transmitted_bytes: int
    transport_ratio: float
    network_time: float
    decode_time: float
    cachegen_enabled: bool
    cachegen_applied: bool
    raw_fallback_layers: int
    kv_cache_dtype: str
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float


def parse_model_arg(raw: str) -> ModelConfig:
    parts = raw.split("|")
    if not parts[0]:
        raise argparse.ArgumentTypeError("model id cannot be empty")

    max_len = int(parts[1]) if len(parts) > 1 and parts[1] else 8192
    dtype = parts[2] if len(parts) > 2 and parts[2] else "bfloat16"
    return ModelConfig(model=parts[0], max_model_len=max_len, dtype=dtype)


def default_models() -> list[ModelConfig]:
    return [
        ModelConfig(
            model=DEFAULT_CLAIM_MODEL,
            max_model_len=8192,
            dtype="bfloat16",
        )
    ]


def get_mode_config(mode: str) -> TransferModeConfig:
    try:
        return MODE_CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark mode: {mode}") from exc


def _dedupe_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _slugify_model(model: str) -> str:
    return model.replace("/", "__")


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value)


def _run_git_command(path: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def ensure_vllm_available() -> None:
    if _VLLM_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "Failed to import vLLM. Install vendored vLLM first. "
        "For Colab or Python-only connector edits, use "
        "`VLLM_USE_PRECOMPILED=1 uv pip install --editable ./third_party/vllm "
        "--torch-backend=auto`."
    ) from _VLLM_IMPORT_ERROR


def ensure_tokenizer_available() -> None:
    if _TOKENIZER_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "Failed to import transformers AutoTokenizer. Install repo requirements "
        "with `pip install -r requirements.txt`."
    ) from _TOKENIZER_IMPORT_ERROR


def load_prompt_corpus(path: Path = PROMPT_CORPUS_PATH) -> str:
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Prompt corpus is empty: {path}")
    return text


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return list(token_ids)


def load_tokenizer(model: str) -> Any:
    ensure_tokenizer_available()
    assert AutoTokenizer is not None
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def build_prompt_specs(
    tokenizer: Any,
    prompt_lengths: Sequence[int],
    *,
    instances_per_length: int = DEFAULT_PROMPT_INSTANCES,
    corpus_text: str | None = None,
) -> list[PromptSpec]:
    unique_lengths = [int(length) for length in _dedupe_preserve_order(prompt_lengths)]
    if not unique_lengths:
        raise ValueError("prompt_lengths must not be empty")
    if any(length <= 0 for length in unique_lengths):
        raise ValueError("prompt_lengths must be positive integers")
    if instances_per_length <= 0:
        raise ValueError("instances_per_length must be positive")

    base_corpus = (corpus_text or load_prompt_corpus()).strip()
    working_text = base_corpus
    token_ids = _encode_text(tokenizer, working_text)

    max_length = max(unique_lengths)
    required_tokens = max_length * (instances_per_length + 1)
    repeat_index = 1
    while len(token_ids) < required_tokens:
        working_text += (
            f"\n\n[Deterministic replay {repeat_index}]\n"
            f"{base_corpus}"
        )
        token_ids = _encode_text(tokenizer, working_text)
        repeat_index += 1

    prompts: list[PromptSpec] = []
    for length in unique_lengths:
        for instance_index in range(instances_per_length):
            offset = instance_index * length
            prompt_token_ids = token_ids[offset : offset + length]
            if len(prompt_token_ids) != length:
                raise ValueError(
                    f"Unable to build a {length}-token prompt at offset {offset}"
                )
            prompts.append(
                PromptSpec(
                    prompt_id=f"len{length}_sample{instance_index + 1}",
                    length_tokens=length,
                    instance_index=instance_index,
                    offset_tokens=offset,
                    prompt_token_ids=prompt_token_ids,
                )
            )

    return prompts


def extract_transfer_stats(
    kv_transfer_params: dict[str, Any] | None,
) -> dict[str, float | int | bool | str]:
    params = kv_transfer_params or {}
    raw_tensor_bytes = _safe_int(params.get("raw_tensor_bytes"), 0)
    if raw_tensor_bytes == 0:
        raw_tensor_bytes = _safe_int(params.get("raw_bytes"), 0)

    transmitted_bytes = _safe_int(params.get("transmitted_bytes"), raw_tensor_bytes)
    if transmitted_bytes == raw_tensor_bytes:
        transmitted_bytes = _safe_int(params.get("compressed_bytes"), transmitted_bytes)
    if transmitted_bytes < 0:
        transmitted_bytes = 0
    if transmitted_bytes == 0 and raw_tensor_bytes > 0:
        transmitted_bytes = raw_tensor_bytes

    if raw_tensor_bytes > 0 and transmitted_bytes > 0:
        default_ratio = raw_tensor_bytes / transmitted_bytes
    else:
        default_ratio = 1.0

    transport_ratio = _safe_float(params.get("transport_ratio"), default_ratio)
    if transport_ratio == default_ratio:
        transport_ratio = _safe_float(params.get("compression_ratio"), transport_ratio)

    network_time = _safe_float(params.get("network_time"), 0.0)
    decode_time = _safe_float(params.get("decode_time"), 0.0)
    cachegen_enabled = _safe_bool(params.get("cachegen_enabled"), False)
    cachegen_applied = _safe_bool(params.get("cachegen_applied"), False)
    raw_fallback_layers = max(0, _safe_int(params.get("raw_fallback_layers"), 0))
    kv_cache_dtype = str(params.get("kv_cache_dtype") or "auto")

    return {
        "raw_tensor_bytes": raw_tensor_bytes,
        "transmitted_bytes": transmitted_bytes,
        "transport_ratio": transport_ratio,
        "network_time": network_time,
        "decode_time": decode_time,
        "cachegen_enabled": cachegen_enabled,
        "cachegen_applied": cachegen_applied,
        "raw_fallback_layers": raw_fallback_layers,
        "kv_cache_dtype": kv_cache_dtype,
        "raw_bytes": raw_tensor_bytes,
        "compressed_bytes": transmitted_bytes,
        "compression_ratio": transport_ratio,
    }


def _build_kv_transfer_config(
    mode: str,
    bandwidth_mbps: float,
    storage_path: Path,
) -> KVTransferConfig:
    mode_config = get_mode_config(mode)
    return KVTransferConfig(
        kv_connector="CacheGenConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cachegen_enabled": mode_config.cachegen_enabled,
            "bandwidth_mbps": float(bandwidth_mbps),
            "shared_storage_path": str(storage_path),
            "compression_level": 3,
        },
    )


def _build_prompt_input(prompt_spec: PromptSpec) -> dict[str, Any]:
    return {"prompt_token_ids": list(prompt_spec.prompt_token_ids)}


def _plain_mode_transfer_stats(mode_config: TransferModeConfig) -> dict[str, float | int | bool | str]:
    return {
        "raw_tensor_bytes": 0,
        "transmitted_bytes": 0,
        "transport_ratio": 1.0,
        "network_time": 0.0,
        "decode_time": 0.0,
        "cachegen_enabled": mode_config.cachegen_enabled,
        "cachegen_applied": False,
        "raw_fallback_layers": 0,
        "kv_cache_dtype": mode_config.kv_cache_dtype,
        "raw_bytes": 0,
        "compressed_bytes": 0,
        "compression_ratio": 1.0,
    }


async def _generate_once(
    engine: AsyncLLMEngine,
    model_label: str,
    mode_label: str,
    prompt_spec: PromptSpec,
    request_suffix: str,
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    submitted_at = time.perf_counter()
    request_id = (
        f"{_slugify_model(model_label)}-{mode_label}-{prompt_spec.prompt_id}"
        f"-{request_suffix}-{int(submitted_at * 1e6)}"
    )
    prompt_input = _build_prompt_input(prompt_spec)

    first_token_at: float | None = None
    final = None

    try:
        stream = engine.generate(
            prompt_input,
            sampling_params,
            request_id=request_id,
        )
    except TypeError:
        try:
            stream = engine.generate(
                prompt_input,
                params=sampling_params,
                request_id=request_id,
            )
        except TypeError:
            stream = engine.generate(prompt_input)

    async for output in stream:
        if first_token_at is None:
            first_token_at = time.perf_counter()
        if output.finished:
            final = output
            break

    end_at = time.perf_counter()

    if final is None:
        raise RuntimeError("vLLM request finished without a final output")
    if not final.outputs:
        raise RuntimeError("vLLM request finished without generated outputs")

    candidate = final.outputs[0]
    prompt_tokens = len(getattr(final, "prompt_token_ids", None) or prompt_spec.prompt_token_ids)
    generated_tokens = len(candidate.token_ids)

    ttft = (first_token_at or end_at) - submitted_at
    total = end_at - submitted_at
    tokens_per_second = generated_tokens / max(total - ttft, 1e-6)
    cached_tokens = _safe_int(
        getattr(final, "num_cached_tokens", getattr(final, "cached_tokens", 0)),
        0,
    )

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "ttft": ttft,
        "total": total,
        "tokens_per_second": tokens_per_second,
        "cached_tokens": cached_tokens,
        "kv_transfer_params": getattr(final, "kv_transfer_params", None),
    }


def validate_cachegen_measured_run(
    cached_tokens: int,
    transfer_stats: dict[str, float | int | bool | str],
) -> None:
    if cached_tokens <= 0:
        raise RuntimeError(
            f"Measured pass for mode=cachegen reported cached_tokens={cached_tokens}; "
            "expected a cache hit."
        )
    if not bool(transfer_stats["cachegen_enabled"]):
        raise RuntimeError("CacheGen mode ran without cachegen_enabled=true in connector stats.")
    if not bool(transfer_stats["cachegen_applied"]):
        raise RuntimeError("CacheGen mode did not apply CacheGen during the measured pass.")
    if int(transfer_stats["raw_fallback_layers"]) > 0:
        raise RuntimeError(
            "CacheGen mode fell back to raw transfer for one or more layers; "
            "claim benchmark results are not trustworthy."
        )


async def _bench_single_prompt_two_pass(
    engine: AsyncLLMEngine,
    model_label: str,
    mode_label: str,
    prompt_spec: PromptSpec,
    repetition: int,
    sampling_params: SamplingParams,
    bandwidth_mbps: float,
    storage_path: Path,
) -> RunResult:
    _clear_dir(storage_path)

    await _generate_once(
        engine=engine,
        model_label=model_label,
        mode_label=mode_label,
        prompt_spec=prompt_spec,
        request_suffix=f"seed-r{repetition}",
        sampling_params=sampling_params,
    )

    measured = await _generate_once(
        engine=engine,
        model_label=model_label,
        mode_label=mode_label,
        prompt_spec=prompt_spec,
        request_suffix=f"measured-r{repetition}",
        sampling_params=sampling_params,
    )

    transfer_stats = extract_transfer_stats(measured["kv_transfer_params"])
    validate_cachegen_measured_run(
        cached_tokens=measured["cached_tokens"],
        transfer_stats=transfer_stats,
    )

    return RunResult(
        model=model_label,
        mode=mode_label,
        repetition=repetition,
        prompt_id=prompt_spec.prompt_id,
        prompt_length_tokens_requested=prompt_spec.length_tokens,
        prompt_length_tokens=measured["prompt_tokens"],
        prompt_instance_index=prompt_spec.instance_index,
        prompt_offset_tokens=prompt_spec.offset_tokens,
        bandwidth_mbps=float(bandwidth_mbps),
        output_tokens=measured["generated_tokens"],
        cached_tokens=measured["cached_tokens"],
        ttft_seconds=measured["ttft"],
        end_to_end_seconds=measured["total"],
        tokens_per_second=measured["tokens_per_second"],
        raw_tensor_bytes=int(transfer_stats["raw_tensor_bytes"]),
        transmitted_bytes=int(transfer_stats["transmitted_bytes"]),
        transport_ratio=float(transfer_stats["transport_ratio"]),
        network_time=float(transfer_stats["network_time"]),
        decode_time=float(transfer_stats["decode_time"]),
        cachegen_enabled=bool(transfer_stats["cachegen_enabled"]),
        cachegen_applied=bool(transfer_stats["cachegen_applied"]),
        raw_fallback_layers=int(transfer_stats["raw_fallback_layers"]),
        kv_cache_dtype=str(transfer_stats["kv_cache_dtype"]),
        raw_bytes=int(transfer_stats["raw_bytes"]),
        compressed_bytes=int(transfer_stats["compressed_bytes"]),
        compression_ratio=float(transfer_stats["compression_ratio"]),
    )


async def _bench_single_prompt_plain(
    engine: AsyncLLMEngine,
    model_label: str,
    mode_config: TransferModeConfig,
    prompt_spec: PromptSpec,
    repetition: int,
    sampling_params: SamplingParams,
    bandwidth_mbps: float,
) -> RunResult:
    measured = await _generate_once(
        engine=engine,
        model_label=model_label,
        mode_label=mode_config.name,
        prompt_spec=prompt_spec,
        request_suffix=f"plain-r{repetition}",
        sampling_params=sampling_params,
    )
    transfer_stats = _plain_mode_transfer_stats(mode_config)

    return RunResult(
        model=model_label,
        mode=mode_config.name,
        repetition=repetition,
        prompt_id=prompt_spec.prompt_id,
        prompt_length_tokens_requested=prompt_spec.length_tokens,
        prompt_length_tokens=measured["prompt_tokens"],
        prompt_instance_index=prompt_spec.instance_index,
        prompt_offset_tokens=prompt_spec.offset_tokens,
        bandwidth_mbps=float(bandwidth_mbps),
        output_tokens=measured["generated_tokens"],
        cached_tokens=measured["cached_tokens"],
        ttft_seconds=measured["ttft"],
        end_to_end_seconds=measured["total"],
        tokens_per_second=measured["tokens_per_second"],
        raw_tensor_bytes=int(transfer_stats["raw_tensor_bytes"]),
        transmitted_bytes=int(transfer_stats["transmitted_bytes"]),
        transport_ratio=float(transfer_stats["transport_ratio"]),
        network_time=float(transfer_stats["network_time"]),
        decode_time=float(transfer_stats["decode_time"]),
        cachegen_enabled=bool(transfer_stats["cachegen_enabled"]),
        cachegen_applied=bool(transfer_stats["cachegen_applied"]),
        raw_fallback_layers=int(transfer_stats["raw_fallback_layers"]),
        kv_cache_dtype=str(transfer_stats["kv_cache_dtype"]),
        raw_bytes=int(transfer_stats["raw_bytes"]),
        compressed_bytes=int(transfer_stats["compressed_bytes"]),
        compression_ratio=float(transfer_stats["compression_ratio"]),
    )


async def _warmup_engine(
    engine: AsyncLLMEngine,
    model_label: str,
    mode_label: str,
    prompt_spec: PromptSpec,
    sampling_params: SamplingParams,
    storage_path: Path,
    clear_storage: bool,
) -> None:
    if clear_storage:
        _clear_dir(storage_path)
    await _generate_once(
        engine=engine,
        model_label=model_label,
        mode_label=mode_label,
        prompt_spec=prompt_spec,
        request_suffix="warmup",
        sampling_params=sampling_params,
    )


async def bench_model_mode_bandwidth(
    cfg: ModelConfig,
    prompts: Sequence[PromptSpec],
    sampling_params: SamplingParams,
    *,
    tensor_parallel_size: int,
    quantization: str | None,
    gpu_memory_utilization: float,
    repeats: int,
    mode: str,
    bandwidth_mbps: float,
    storage_path: Path,
) -> list[RunResult]:
    ensure_vllm_available()
    mode_config = get_mode_config(mode)
    engine_kwargs: dict[str, Any] = {
        "model": cfg.model,
        "max_model_len": cfg.max_model_len,
        "dtype": cfg.dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "quantization": quantization,
        "kv_cache_dtype": mode_config.kv_cache_dtype,
        "enable_prefix_caching": False,
        "enforce_eager": True,
    }
    if mode_config.uses_kv_transfer:
        engine_kwargs["kv_transfer_config"] = _build_kv_transfer_config(
            mode=mode,
            bandwidth_mbps=bandwidth_mbps,
            storage_path=storage_path,
        )
        engine_kwargs["disable_hybrid_kv_cache_manager"] = True

    args = AsyncEngineArgs(
        **engine_kwargs,
    )

    engine = AsyncLLMEngine.from_engine_args(args)
    results: list[RunResult] = []

    try:
        warmup_prompt = min(prompts, key=lambda prompt: (prompt.length_tokens, prompt.instance_index))
        await _warmup_engine(
            engine=engine,
            model_label=cfg.model,
            mode_label=mode,
            prompt_spec=warmup_prompt,
            sampling_params=sampling_params,
            storage_path=storage_path,
            clear_storage=mode_config.uses_kv_transfer,
        )

        for repetition in range(1, repeats + 1):
            for prompt_spec in prompts:
                if mode_config.uses_kv_transfer:
                    result = await _bench_single_prompt_two_pass(
                        engine=engine,
                        model_label=cfg.model,
                        mode_label=mode,
                        prompt_spec=prompt_spec,
                        repetition=repetition,
                        sampling_params=sampling_params,
                        bandwidth_mbps=bandwidth_mbps,
                        storage_path=storage_path,
                    )
                else:
                    result = await _bench_single_prompt_plain(
                        engine=engine,
                        model_label=cfg.model,
                        mode_config=mode_config,
                        prompt_spec=prompt_spec,
                        repetition=repetition,
                        sampling_params=sampling_params,
                        bandwidth_mbps=bandwidth_mbps,
                    )
                results.append(result)
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown:
            maybe_coro = shutdown()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

    return results


def capture_environment_meta() -> dict[str, Any]:
    now = datetime.now().astimezone()
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "timestamp": now.timestamp(),
        "generated_at_iso": now.isoformat(),
        "python_version": sys.version.split()[0],
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": gpu_name,
        "vllm_version": getattr(vllm, "__version__", None) if vllm is not None else None,
        "vllm_submodule_commit": _run_git_command(VENDORED_VLLM, "rev-parse", "HEAD")
        if VENDORED_VLLM.exists()
        else None,
        "benchmark_git_commit": _run_git_command(SCRIPT_DIR, "rev-parse", "HEAD"),
        "vllm_import_error": str(_VLLM_IMPORT_ERROR) if _VLLM_IMPORT_ERROR else None,
    }


def _median_summary(rows: Sequence[RunResult], *, include_prompt_length: bool) -> dict[str, Any]:
    first = rows[0]
    summary = {
        "model": first.model,
        "mode": first.mode,
        "bandwidth_mbps": first.bandwidth_mbps,
        "num_runs": len(rows),
        "repetitions": len({row.repetition for row in rows}),
        "prompt_instances": len({row.prompt_id for row in rows}),
        "median_ttft_seconds": median(row.ttft_seconds for row in rows),
        "median_end_to_end_seconds": median(row.end_to_end_seconds for row in rows),
        "median_tokens_per_second": median(row.tokens_per_second for row in rows),
        "median_raw_tensor_bytes": median(row.raw_tensor_bytes for row in rows),
        "median_transmitted_bytes": median(row.transmitted_bytes for row in rows),
        "median_transport_ratio": median(row.transport_ratio for row in rows),
        "median_network_time": median(row.network_time for row in rows),
        "median_decode_time": median(row.decode_time for row in rows),
    }
    if include_prompt_length:
        summary["prompt_length_tokens_requested"] = first.prompt_length_tokens_requested
    return summary


def summarize_runs(runs: Sequence[RunResult]) -> dict[str, list[dict[str, Any]]]:
    by_mode_prompt_length: dict[tuple[str, str, float, int], list[RunResult]] = {}
    by_mode: dict[tuple[str, str, float], list[RunResult]] = {}

    for row in runs:
        by_mode_prompt_length.setdefault(
            (row.model, row.mode, row.bandwidth_mbps, row.prompt_length_tokens_requested),
            [],
        ).append(row)
        by_mode.setdefault((row.model, row.mode, row.bandwidth_mbps), []).append(row)

    prompt_length_summaries = [
        _median_summary(group, include_prompt_length=True)
        for _, group in sorted(by_mode_prompt_length.items())
    ]
    mode_summaries = [
        _median_summary(group, include_prompt_length=False)
        for _, group in sorted(by_mode.items())
    ]

    return {
        "by_mode_prompt_length": prompt_length_summaries,
        "by_mode": mode_summaries,
    }


def build_claim_check(
    runs: Sequence[RunResult],
    *,
    claim_model: str = DEFAULT_CLAIM_MODEL,
    bandwidth_mbps: float = CLAIM_TARGET_BANDWIDTH_MBPS,
) -> dict[str, Any]:
    claim_runs = [
        row
        for row in runs
        if row.model == claim_model and float(row.bandwidth_mbps) == float(bandwidth_mbps)
    ]
    quantized_fp8_runs = [row for row in claim_runs if row.mode == "quantized_fp8"]
    cachegen_runs = [row for row in claim_runs if row.mode == "cachegen"]

    base = {
        "evaluated": False,
        "model": claim_model,
        "bandwidth_mbps": float(bandwidth_mbps),
        "metric": "median_ttft_quantized_fp8 / median_ttft_cachegen",
        "paper_range": {
            "min_speedup": CLAIM_SPEEDUP_RANGE[0],
            "max_speedup": CLAIM_SPEEDUP_RANGE[1],
        },
        "range_status": "not_evaluated",
        "range_assessment": "claim run not evaluated",
        "ttft_speedup_3gbps": None,
        "median_ttft_quantized_fp8": None,
        "median_ttft_cachegen": None,
        "per_prompt_length": [],
    }

    if not quantized_fp8_runs or not cachegen_runs:
        base["reason"] = (
            "Need both quantized_fp8 and cachegen runs for the claim model at 3000 Mbps."
        )
        return base

    quantized_fp8_ttft = median(row.ttft_seconds for row in quantized_fp8_runs)
    cachegen_ttft = median(row.ttft_seconds for row in cachegen_runs)
    if cachegen_ttft <= 0:
        base["reason"] = "CacheGen median TTFT was non-positive."
        return base

    speedup = quantized_fp8_ttft / cachegen_ttft
    if speedup < (CLAIM_SPEEDUP_RANGE[0] - CLAIM_TOLERANCE):
        range_status = "below_paper_range"
        range_assessment = "below the paper range"
    elif speedup <= (CLAIM_SPEEDUP_RANGE[1] + CLAIM_TOLERANCE):
        range_status = "within_paper_range"
        range_assessment = "falls within 3.2-3.7x"
    else:
        range_status = "above_paper_range"
        range_assessment = "exceeds the paper range"

    per_prompt_length: list[dict[str, Any]] = []
    for prompt_length in sorted({row.prompt_length_tokens_requested for row in claim_runs}):
        per_length_q = [
            row.ttft_seconds
            for row in quantized_fp8_runs
            if row.prompt_length_tokens_requested == prompt_length
        ]
        per_length_c = [
            row.ttft_seconds
            for row in cachegen_runs
            if row.prompt_length_tokens_requested == prompt_length
        ]
        if not per_length_q or not per_length_c:
            continue
        median_q = median(per_length_q)
        median_c = median(per_length_c)
        per_prompt_length.append(
            {
                "prompt_length_tokens_requested": prompt_length,
                "median_ttft_quantized_fp8": median_q,
                "median_ttft_cachegen": median_c,
                "speedup": median_q / median_c if median_c > 0 else None,
            }
        )

    return {
        **base,
        "evaluated": True,
        "range_status": range_status,
        "range_assessment": range_assessment,
        "ttft_speedup_3gbps": speedup,
        "median_ttft_quantized_fp8": quantized_fp8_ttft,
        "median_ttft_cachegen": cachegen_ttft,
        "per_prompt_length": per_prompt_length,
    }


def serialize_config(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "models": [asdict(model) for model in config.models],
        "modes": list(config.modes),
        "bandwidth_mbps": list(config.bandwidth_mbps),
        "prompt_lengths": list(config.prompt_lengths),
        "prompt_instances_per_length": config.prompt_instances_per_length,
        "repeats": config.repeats,
        "max_tokens": config.max_tokens,
        "stop": list(config.stop),
        "tensor_parallel_size": config.tensor_parallel_size,
        "quantization": config.quantization,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "cache_root": str(config.cache_root),
        "output_path": str(config.output_path),
        "prompt_corpus_path": str(config.prompt_corpus_path),
        "claim_model": config.claim_model,
        "claim_bandwidth_mbps": config.claim_bandwidth_mbps,
        "connector": "CacheGenConnector",
    }


async def run_benchmark_suite(config: BenchmarkConfig) -> dict[str, Any]:
    ensure_vllm_available()
    ensure_tokenizer_available()

    cache_root = config.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    corpus_text = load_prompt_corpus(config.prompt_corpus_path)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=config.max_tokens,
        stop=list(config.stop),
    )

    all_results: list[RunResult] = []
    for model_config in config.models:
        tokenizer = load_tokenizer(model_config.model)
        prompts = build_prompt_specs(
            tokenizer,
            config.prompt_lengths,
            instances_per_length=config.prompt_instances_per_length,
            corpus_text=corpus_text,
        )

        model_slug = _slugify_model(model_config.model)
        for mode in config.modes:
            for bandwidth_mbps in config.bandwidth_mbps:
                storage_path = cache_root / model_slug / mode / f"bw_{bandwidth_mbps:g}"
                results = await bench_model_mode_bandwidth(
                    cfg=model_config,
                    prompts=prompts,
                    sampling_params=sampling_params,
                    tensor_parallel_size=config.tensor_parallel_size,
                    quantization=config.quantization,
                    gpu_memory_utilization=config.gpu_memory_utilization,
                    repeats=config.repeats,
                    mode=mode,
                    bandwidth_mbps=bandwidth_mbps,
                    storage_path=storage_path,
                )
                all_results.extend(results)

    meta = capture_environment_meta()
    summaries = summarize_runs(all_results)
    claim_check = build_claim_check(
        all_results,
        claim_model=config.claim_model,
        bandwidth_mbps=config.claim_bandwidth_mbps,
    )

    return {
        "meta": meta,
        "config": serialize_config(config),
        "runs": [asdict(result) for result in all_results],
        "summaries": summaries,
        "claim_check": claim_check,
    }


def save_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return output_path
