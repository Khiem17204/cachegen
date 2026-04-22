"""Offline compression-ratio and perplexity-delta benchmark."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from decoder import CacheGenDecoder
from encoder import CacheGenEncoder
from encoder.wire_format import deserialize_encoded_chunks, serialize_encoded_chunks
from kv_extraction_hf.cache_conversion import (
    past_key_values_to_tensor,
    tensor_to_past_key_values,
)

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "benchmarks" / "offline_subset_results.json"


@dataclass(frozen=True)
class OfflineBenchmarkConfig:
    model: str = "gpt2"
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"
    num_windows: int = 64
    window_length: int = 192
    cached_prefix_length: int = 127
    scored_continuation_length: int = 64
    chunk_size: int = 64
    compression_level: int = 3
    output_path: Path = DEFAULT_OUTPUT_PATH
    device: str = "cpu"


@dataclass
class WindowMetrics:
    window_index: int
    token_offset: int
    cached_prefix_tokens: int
    scored_tokens: int
    raw_tensor_bytes: int
    cachegen_payload_bytes: int
    compression_ratio: float
    baseline_nll: float
    cachegen_nll: float
    avg_nll_baseline: float
    avg_nll_cachegen: float
    ppl_baseline: float
    ppl_cachegen: float
    ppl_delta_abs: float
    ppl_delta_pct: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline CacheGen compression/perplexity subset benchmark"
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-windows", type=int, default=64)
    parser.add_argument("--window-length", type=int, default=192)
    parser.add_argument("--cached-prefix-length", type=int, default=127)
    parser.add_argument("--scored-continuation-length", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", default="cpu")
    return parser


def run_offline_benchmark(config: OfflineBenchmarkConfig) -> dict[str, Any]:
    validate_config(config)

    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Offline benchmarks require `datasets` and `transformers`. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = AutoModelForCausalLM.from_pretrained(config.model)
    model.eval()
    model.to(config.device)

    dataset = load_dataset(config.dataset, config.dataset_config, split=config.split)
    token_ids = build_dataset_token_ids(tokenizer, dataset)
    windows = build_token_windows(
        token_ids,
        num_windows=config.num_windows,
        window_length=config.window_length,
    )
    return evaluate_windows(model, windows, config)


def evaluate_windows(
    model: Any,
    windows: list[list[int]],
    config: OfflineBenchmarkConfig,
) -> dict[str, Any]:
    validate_config(config)
    encoder = CacheGenEncoder(
        chunk_size=config.chunk_size,
        compression_level=config.compression_level,
    )
    decoder = CacheGenDecoder()

    rows = [
        score_window(
            model,
            window,
            window_index=index,
            token_offset=index * config.window_length,
            config=config,
            encoder=encoder,
            decoder=decoder,
        )
        for index, window in enumerate(windows)
    ]
    return build_report(config, rows)


def score_window(
    model: Any,
    token_ids: list[int],
    *,
    window_index: int,
    token_offset: int,
    config: OfflineBenchmarkConfig,
    encoder: CacheGenEncoder,
    decoder: CacheGenDecoder,
) -> WindowMetrics:
    if len(token_ids) != config.window_length:
        raise ValueError(
            f"Expected {config.window_length} tokens, got window with {len(token_ids)}"
        )

    device = torch.device(config.device)
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    prefix_end = config.cached_prefix_length
    score_end = prefix_end + config.scored_continuation_length

    context_ids = ids[:, :prefix_end]
    score_input_ids = ids[:, prefix_end:score_end]
    target_ids = ids[:, prefix_end + 1 : score_end + 1]

    with torch.no_grad():
        context_outputs = model(context_ids, use_cache=True)
        original_past = context_outputs.past_key_values
        kv_tensor = past_key_values_to_tensor(original_past, dtype=None, device="cpu")

        baseline_outputs = model(
            score_input_ids,
            past_key_values=original_past,
            use_cache=True,
        )
        baseline_nll = token_nll_sum(baseline_outputs.logits, target_ids)

        encoded_chunks = encoder.encode(kv_tensor)
        payload = serialize_encoded_chunks(encoded_chunks)
        reconstructed = decoder.decode(deserialize_encoded_chunks(payload))
        reconstructed_past = tensor_to_past_key_values(
            reconstructed,
            device=device,
            dtype=kv_tensor.dtype,
            like=original_past,
            model_config=getattr(model, "config", None),
        )
        cachegen_outputs = model(
            score_input_ids,
            past_key_values=reconstructed_past,
            use_cache=True,
        )
        cachegen_nll = token_nll_sum(cachegen_outputs.logits, target_ids)

    raw_tensor_bytes = int(kv_tensor.numel() * kv_tensor.element_size())
    payload_bytes = len(payload)
    scored_tokens = int(target_ids.numel())
    compression_ratio = raw_tensor_bytes / max(payload_bytes, 1)
    avg_baseline = baseline_nll / scored_tokens
    avg_cachegen = cachegen_nll / scored_tokens
    ppl_baseline = math.exp(avg_baseline)
    ppl_cachegen = math.exp(avg_cachegen)
    delta_abs = ppl_cachegen - ppl_baseline
    delta_pct = (delta_abs / ppl_baseline) * 100.0 if ppl_baseline else math.inf

    return WindowMetrics(
        window_index=window_index,
        token_offset=token_offset,
        cached_prefix_tokens=config.cached_prefix_length,
        scored_tokens=scored_tokens,
        raw_tensor_bytes=raw_tensor_bytes,
        cachegen_payload_bytes=payload_bytes,
        compression_ratio=compression_ratio,
        baseline_nll=baseline_nll,
        cachegen_nll=cachegen_nll,
        avg_nll_baseline=avg_baseline,
        avg_nll_cachegen=avg_cachegen,
        ppl_baseline=ppl_baseline,
        ppl_cachegen=ppl_cachegen,
        ppl_delta_abs=delta_abs,
        ppl_delta_pct=delta_pct,
    )


def token_nll_sum(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        target_ids.reshape(-1),
        reduction="sum",
    )
    return float(loss.item())


def build_report(
    config: OfflineBenchmarkConfig,
    rows: list[WindowMetrics],
) -> dict[str, Any]:
    total_tokens = sum(row.scored_tokens for row in rows)
    if total_tokens <= 0:
        raise ValueError("Cannot build report without scored tokens")

    total_baseline_nll = sum(row.baseline_nll for row in rows)
    total_cachegen_nll = sum(row.cachegen_nll for row in rows)
    raw_bytes = sum(row.raw_tensor_bytes for row in rows)
    payload_bytes = sum(row.cachegen_payload_bytes for row in rows)
    avg_baseline = total_baseline_nll / total_tokens
    avg_cachegen = total_cachegen_nll / total_tokens
    ppl_baseline = math.exp(avg_baseline)
    ppl_cachegen = math.exp(avg_cachegen)
    delta_abs = ppl_cachegen - ppl_baseline
    delta_pct = (delta_abs / ppl_baseline) * 100.0 if ppl_baseline else math.inf

    config_dict = asdict(config)
    config_dict["output_path"] = str(config.output_path)

    return {
        "benchmark": "offline_cachegen_subset",
        "config": config_dict,
        "aggregates": {
            "windows": len(rows),
            "total_scored_tokens": total_tokens,
            "raw_tensor_bytes": raw_bytes,
            "cachegen_payload_bytes": payload_bytes,
            "compression_ratio": raw_bytes / max(payload_bytes, 1),
            "avg_nll_baseline": avg_baseline,
            "avg_nll_cachegen": avg_cachegen,
            "ppl_baseline": ppl_baseline,
            "ppl_cachegen": ppl_cachegen,
            "ppl_delta_abs": delta_abs,
            "ppl_delta_pct": delta_pct,
        },
        "windows": [asdict(row) for row in rows],
    }


def build_dataset_token_ids(tokenizer: Any, dataset: Any) -> list[int]:
    texts = [row["text"].strip() for row in dataset if row.get("text", "").strip()]
    if not texts:
        raise ValueError("Dataset split did not contain any non-empty text rows")
    encoded = tokenizer(
        "\n\n".join(texts),
        add_special_tokens=False,
        return_attention_mask=False,
        verbose=False,
    )
    return list(encoded["input_ids"])


def build_token_windows(
    token_ids: list[int],
    *,
    num_windows: int,
    window_length: int,
) -> list[list[int]]:
    required = num_windows * window_length
    if len(token_ids) < required:
        raise ValueError(f"Need {required} tokens, only found {len(token_ids)}")
    return [
        token_ids[offset : offset + window_length]
        for offset in range(0, required, window_length)
    ]


def validate_config(config: OfflineBenchmarkConfig) -> None:
    if config.num_windows <= 0:
        raise ValueError("num_windows must be positive")
    if config.window_length <= 0:
        raise ValueError("window_length must be positive")
    if config.cached_prefix_length <= 0:
        raise ValueError("cached_prefix_length must be positive")
    if config.scored_continuation_length <= 0:
        raise ValueError("scored_continuation_length must be positive")
    expected = config.cached_prefix_length + config.scored_continuation_length + 1
    if config.window_length != expected:
        raise ValueError(
            "window_length must equal cached_prefix_length + "
            f"scored_continuation_length + 1, got {config.window_length} != {expected}"
        )


def save_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    args = build_arg_parser().parse_args()
    config = OfflineBenchmarkConfig(
        model=args.model,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        num_windows=args.num_windows,
        window_length=args.window_length,
        cached_prefix_length=args.cached_prefix_length,
        scored_continuation_length=args.scored_continuation_length,
        chunk_size=args.chunk_size,
        compression_level=args.compression_level,
        output_path=args.output,
        device=args.device,
    )
    report = run_offline_benchmark(config)
    out_path = save_report(report, config.output_path)
    aggregates = report["aggregates"]
    print(f"Saved offline benchmark report to {out_path}")
    print(f"Compression ratio: {aggregates['compression_ratio']:.3f}x")
    print(
        "Perplexity: "
        f"{aggregates['ppl_baseline']:.3f} baseline, "
        f"{aggregates['ppl_cachegen']:.3f} CacheGen "
        f"({aggregates['ppl_delta_pct']:+.2f}%)"
    )


if __name__ == "__main__":
    main()
