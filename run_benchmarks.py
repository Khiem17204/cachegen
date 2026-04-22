"""CLI entrypoint for the Colab-ready TTFT reproduction benchmark."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ttft_benchmark import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_RESULTS_PATH,
    MODE_CONFIGS,
    BenchmarkConfig,
    default_models,
    extract_transfer_stats as _extract_transfer_stats,
    parse_model_arg,
    run_benchmark_suite,
    save_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the TTFT reproduction benchmark with vendored vLLM"
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
        "--prompt-lengths",
        nargs="+",
        type=int,
        default=[2048, 4096],
        help="Tokenizer-exact prompt lengths to benchmark",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of measured repetitions per prompt instance",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Max new tokens to generate per prompt",
    )
    parser.add_argument(
        "--stop",
        nargs="*",
        default=[],
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
        help="Optional model-weight quantization mode (not the paper comparator)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization target for vLLM",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_CONFIGS),
        default=["quantized_fp8", "cachegen"],
        help="Transfer modes to benchmark",
    )
    parser.add_argument(
        "--bandwidth-mbps",
        nargs="+",
        type=float,
        default=[3000.0],
        help="Bandwidth values (Mbps) to apply as real throttling during KV transfer",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(DEFAULT_CACHE_ROOT),
        help="Directory used by CacheGenConnector for external KV storage",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to write the benchmark report JSON",
    )
    parser.add_argument(
        "--prompt-corpus",
        type=str,
        default=None,
        help="Optional prompt corpus text file",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.models:
        args.models = default_models()

    args.modes = list(dict.fromkeys(args.modes))
    args.bandwidth_mbps = list(dict.fromkeys(args.bandwidth_mbps))
    args.prompt_lengths = list(dict.fromkeys(args.prompt_lengths))

    if any(bw <= 0 for bw in args.bandwidth_mbps):
        parser.error("all --bandwidth-mbps values must be positive")
    if any(length <= 0 for length in args.prompt_lengths):
        parser.error("all --prompt-lengths values must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be within (0, 1]")

    config = BenchmarkConfig(
        models=args.models,
        modes=args.modes,
        bandwidth_mbps=args.bandwidth_mbps,
        prompt_lengths=args.prompt_lengths,
        repeats=args.repeats,
        max_tokens=args.max_tokens,
        stop=args.stop,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cache_root=Path(args.cache_root),
        output_path=Path(args.output),
        prompt_corpus_path=(
            Path(args.prompt_corpus) if args.prompt_corpus else DEFAULT_RESULTS_PATH.parent / "prompt_corpus.txt"
        ),
    )

    report = asyncio.run(run_benchmark_suite(config))
    out_path = save_report(report, config.output_path)
    print(f"Saved TTFT benchmark report to {out_path}")


if __name__ == "__main__":
    main()
