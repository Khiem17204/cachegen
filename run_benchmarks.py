"""Live vLLM benchmark harness with true end-to-end CacheGen connector runs.

This script benchmarks two transfer modes on real vLLM generation:
- baseline: raw KV transfer through CacheGenConnector (`cachegen_enabled=False`)
- cachegen: compressed KV transfer through CacheGenConnector (`cachegen_enabled=True`)

For each prompt we run two passes:
1) seed pass: generate once to populate external KV storage
2) measured pass: generate again to load KV from external storage and record metrics

The measured pass provides TTFT/end-to-end/throughput from real generation and
transfer stats from connector `kv_transfer_params`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Prefer vendored vLLM source if present.
SCRIPT_DIR = Path(__file__).resolve().parent
VENDORED_VLLM = SCRIPT_DIR / "third_party" / "vllm"
if VENDORED_VLLM.exists():
    sys.path.insert(0, str(VENDORED_VLLM))

_VLLM_IMPORT_ERROR: Exception | None = None
try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.config import KVTransferConfig
except Exception as exc:  # pragma: no cover - import environment dependent
    AsyncEngineArgs = AsyncLLMEngine = SamplingParams = KVTransferConfig = None
    _VLLM_IMPORT_ERROR = exc

PROMPTS: Dict[str, str] = {
    "short": "What is the capital of France?",
    "medium": (
        "Explain how transformers leverage self-attention to model long-range "
        "dependencies in language. Include an example of positional encoding and "
        "discuss its effect on downstream tasks."
    ),
    "long": (
        "Write an in-depth summary of how reinforcement learning from human "
        "feedback (RLHF) is used to align large language models, covering "
        "preference collection, reward model training, policy optimisation, "
        "safety mitigations, and evaluation challenges across multilingual "
        "deployments."
    ),
}


@dataclass
class ModelConfig:
    model: str
    max_model_len: int = 4096
    dtype: str = "bfloat16"


@dataclass
class RunResult:
    model: str
    mode: str
    prompt_name: str
    bandwidth_mbps: float
    prompt_len_tokens: int
    output_tokens: int
    cached_tokens: int
    ttft_seconds: float
    end_to_end_seconds: float
    tokens_per_second: float
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    network_time: float
    decode_time: float


def parse_model_arg(raw: str) -> ModelConfig:
    parts = raw.split("|")
    if not parts[0]:
        raise argparse.ArgumentTypeError("model id cannot be empty")

    max_len = int(parts[1]) if len(parts) > 1 and parts[1] else 4096
    dtype = parts[2] if len(parts) > 2 and parts[2] else "bfloat16"
    return ModelConfig(model=parts[0], max_model_len=max_len, dtype=dtype)


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


def _extract_transfer_stats(kv_transfer_params: dict[str, Any] | None) -> dict[str, float | int]:
    params = kv_transfer_params or {}
    raw_bytes = _safe_int(params.get("raw_bytes"), 0)
    compressed_bytes = _safe_int(params.get("compressed_bytes"), raw_bytes)
    if compressed_bytes < 0:
        compressed_bytes = 0
    if compressed_bytes == 0 and raw_bytes > 0:
        compressed_bytes = raw_bytes

    if compressed_bytes > 0 and raw_bytes > 0:
        default_ratio = raw_bytes / compressed_bytes
    else:
        default_ratio = 1.0
    compression_ratio = _safe_float(params.get("compression_ratio"), default_ratio)
    network_time = _safe_float(params.get("network_time"), 0.0)
    decode_time = _safe_float(params.get("decode_time"), 0.0)

    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": compression_ratio,
        "network_time": network_time,
        "decode_time": decode_time,
    }


def _build_kv_transfer_config(
    mode: str,
    bandwidth_mbps: float,
    storage_path: Path,
) -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="CacheGenConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cachegen_enabled": mode == "cachegen",
            "bandwidth_mbps": float(bandwidth_mbps),
            "shared_storage_path": str(storage_path),
            "compression_level": 3,
        },
    )


async def _generate_once(
    engine: AsyncLLMEngine,
    model_label: str,
    mode_label: str,
    prompt_name: str,
    prompt: str,
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    submitted_at = time.perf_counter()
    request_id = (
        f"{model_label}-{mode_label}-{prompt_name}-{int(submitted_at * 1e6)}"
    )

    first_token_at: Optional[float] = None
    final = None

    try:
        stream = engine.generate(prompt, sampling_params, request_id=request_id)
    except TypeError:
        try:
            stream = engine.generate(prompt, params=sampling_params, request_id=request_id)
        except TypeError:
            stream = engine.generate(prompt)

    async for output in stream:
        if first_token_at is None:
            first_token_at = time.perf_counter()
        if output.finished:
            final = output
            break

    end_at = time.perf_counter()

    if final is None:
        raise RuntimeError("vLLM request finished without a final output")

    candidate = final.outputs[0]
    prompt_tokens = len(final.prompt_token_ids)
    generated_tokens = len(candidate.token_ids)

    ttft = (first_token_at or end_at) - submitted_at
    total = end_at - submitted_at
    tokens_per_s = generated_tokens / max(total - ttft, 1e-6)

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "ttft": ttft,
        "total": total,
        "tokens_per_s": tokens_per_s,
        "kv_transfer_params": getattr(final, "kv_transfer_params", None),
        "cached_tokens": _safe_int(getattr(final, "num_cached_tokens", 0), 0),
    }


async def _bench_single_prompt_two_pass(
    engine: AsyncLLMEngine,
    model_label: str,
    mode_label: str,
    prompt_name: str,
    prompt: str,
    sampling_params: SamplingParams,
    bandwidth_mbps: float,
    storage_path: Path,
) -> RunResult:
    # Ensure every prompt run starts from a clean external cache.
    _clear_dir(storage_path)

    # Seed pass populates external KV storage.
    await _generate_once(
        engine=engine,
        model_label=model_label,
        mode_label=mode_label,
        prompt_name=f"{prompt_name}-seed",
        prompt=prompt,
        sampling_params=sampling_params,
    )

    # Measured pass reuses stored KV and reports transfer stats.
    measured = await _generate_once(
        engine=engine,
        model_label=model_label,
        mode_label=mode_label,
        prompt_name=f"{prompt_name}-measured",
        prompt=prompt,
        sampling_params=sampling_params,
    )

    transfer_stats = _extract_transfer_stats(measured["kv_transfer_params"])

    return RunResult(
        model=model_label,
        mode=mode_label,
        prompt_name=prompt_name,
        bandwidth_mbps=float(bandwidth_mbps),
        prompt_len_tokens=measured["prompt_tokens"],
        output_tokens=measured["generated_tokens"],
        cached_tokens=measured["cached_tokens"],
        ttft_seconds=measured["ttft"],
        end_to_end_seconds=measured["total"],
        tokens_per_second=measured["tokens_per_s"],
        raw_bytes=int(transfer_stats["raw_bytes"]),
        compressed_bytes=int(transfer_stats["compressed_bytes"]),
        compression_ratio=float(transfer_stats["compression_ratio"]),
        network_time=float(transfer_stats["network_time"]),
        decode_time=float(transfer_stats["decode_time"]),
    )


async def bench_model_mode_bandwidth(
    cfg: ModelConfig,
    sampling_params: SamplingParams,
    tp_size: int,
    quantization: Optional[str],
    mode: str,
    bandwidth_mbps: float,
    storage_path: Path,
) -> List[RunResult]:
    kv_transfer_config = _build_kv_transfer_config(
        mode=mode,
        bandwidth_mbps=bandwidth_mbps,
        storage_path=storage_path,
    )

    args = AsyncEngineArgs(
        model=cfg.model,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        quantization=quantization,
        kv_transfer_config=kv_transfer_config,
        enable_prefix_caching=False,
        disable_hybrid_kv_cache_manager=True,
        enforce_eager=True,
    )

    engine = AsyncLLMEngine.from_engine_args(args)
    results: List[RunResult] = []

    try:
        for prompt_name, prompt in PROMPTS.items():
            result = await _bench_single_prompt_two_pass(
                engine=engine,
                model_label=cfg.model,
                mode_label=mode,
                prompt_name=prompt_name,
                prompt=prompt,
                sampling_params=sampling_params,
                bandwidth_mbps=bandwidth_mbps,
                storage_path=storage_path,
            )
            results.append(result)
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown:
            maybe_coro = shutdown()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

    return results


async def bench_model(
    cfg: ModelConfig,
    sampling_params: SamplingParams,
    tp_size: int,
    quantization: Optional[str],
    modes: list[str],
    bandwidths: list[float],
    cache_root: Path,
) -> List[RunResult]:
    all_results: List[RunResult] = []
    model_slug = _slugify_model(cfg.model)

    for mode in modes:
        for bandwidth_mbps in bandwidths:
            storage_path = cache_root / model_slug / mode / f"bw_{bandwidth_mbps:g}"
            results = await bench_model_mode_bandwidth(
                cfg=cfg,
                sampling_params=sampling_params,
                tp_size=tp_size,
                quantization=quantization,
                mode=mode,
                bandwidth_mbps=bandwidth_mbps,
                storage_path=storage_path,
            )
            all_results.extend(results)

    return all_results


async def main_async(args: argparse.Namespace) -> None:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=[] if args.stop is None else args.stop,
    )

    cache_root = Path(args.cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    all_results: List[RunResult] = []
    for cfg in args.models:
        model_results = await bench_model(
            cfg=cfg,
            sampling_params=sampling_params,
            tp_size=args.tensor_parallel_size,
            quantization=args.quantization,
            modes=args.modes,
            bandwidths=args.bandwidth_mbps,
            cache_root=cache_root,
        )
        all_results.extend(model_results)

    now = time.time()
    report = {
        "generated_at": now,
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "sampling_params": {
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": [] if args.stop is None else args.stop,
        },
        "models": [asdict(m) for m in args.models],
        "tensor_parallel_size": args.tensor_parallel_size,
        "quantization": args.quantization,
        "modes": args.modes,
        "bandwidth_mbps": args.bandwidth_mbps,
        "cache_root": str(cache_root),
        "connector": "CacheGenConnector",
        "runs": [asdict(r) for r in all_results],
    }

    out_path = SCRIPT_DIR / "benchmarks" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved live benchmark results to {out_path} at {report['generated_at_iso']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run true end-to-end vLLM baseline vs CacheGen benchmarks"
    )
    parser.add_argument(
        "--model",
        "--models",
        dest="models",
        action="append",
        type=parse_model_arg,
        help="Model spec as model_id|max_len|dtype; can be passed multiple times",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max new tokens to generate per prompt",
    )
    parser.add_argument(
        "--stop",
        nargs="*",
        default=None,
        help="Optional stop sequences",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Optional vLLM quantization mode (awq/gptq/fp8/etc.)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "cachegen"],
        default=["baseline", "cachegen"],
        help="Transfer modes to benchmark",
    )
    parser.add_argument(
        "--bandwidth-mbps",
        nargs="+",
        type=float,
        default=[1000.0],
        help="Bandwidth values (Mbps) to apply as real throttling during KV transfer",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(SCRIPT_DIR / "benchmarks" / "cache_store"),
        help="Directory used by CacheGenConnector for external KV storage",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if _VLLM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import vLLM. Install vendored vLLM first with "
            "`pip install -e ./third_party/vllm`."
        ) from _VLLM_IMPORT_ERROR

    if not args.models:
        args.models = [
            ModelConfig(model="facebook/opt-125m", max_model_len=2048, dtype="float16"),
            ModelConfig(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                max_model_len=8192,
                dtype="bfloat16",
            ),
        ]

    args.modes = list(dict.fromkeys(args.modes))
    args.bandwidth_mbps = list(dict.fromkeys(args.bandwidth_mbps))

    if any(bw <= 0 for bw in args.bandwidth_mbps):
        parser.error("all --bandwidth-mbps values must be positive")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
