"""Live vLLM benchmark harness (Stage 5).

Runs real generations through vLLM on a GPU and records TTFT, end‑to‑end
latency, and throughput across a prompt suite and one or more models.

Usage examples
--------------

Default (opt-125m + Meta-Llama-3-8B-Instruct):
    python run_benchmarks.py

Custom models (model_id|max_len|dtype):
    python run_benchmarks.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct|8192|bfloat16 \\
        --model mistralai/Mistral-7B-Instruct-v0.3|8192|bfloat16

70B with 2×A100 or equivalent:
    python run_benchmarks.py \\
        --model meta-llama/Llama-2-70b-chat-hf|4096|bfloat16 \\
        --tensor-parallel-size 2 \\
        --quantization awq

Outputs
-------
Writes benchmarks/results.json with per-run metrics. Use visualize_results.py
to generate PNGs from the JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# Prompt suite (aligns with previous simulated harness for comparison)
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
    prompt_len_tokens: int
    output_tokens: int
    ttft_seconds: float
    end_to_end_seconds: float
    tokens_per_second: float


def parse_model_arg(raw: str) -> ModelConfig:
    """Parse ``model|max_len|dtype`` into a ModelConfig."""

    parts = raw.split("|")
    if not parts[0]:
        raise argparse.ArgumentTypeError("model id cannot be empty")

    max_len = int(parts[1]) if len(parts) > 1 and parts[1] else 4096
    dtype = parts[2] if len(parts) > 2 and parts[2] else "bfloat16"
    return ModelConfig(model=parts[0], max_model_len=max_len, dtype=dtype)


async def _bench_single_prompt(
    engine: AsyncLLMEngine,
    model_label: str,
    mode_label: str,
    prompt_name: str,
    prompt: str,
    sampling_params: SamplingParams,
) -> RunResult:
    submitted_at = time.perf_counter()
    request_id = f"{model_label}-{mode_label}-{prompt_name}-{int(submitted_at * 1e6)}"

    # Use streaming generate(); adapt to signature differences across vLLM versions.
    first_token_at: Optional[float] = None
    final = None

    try:
        stream = engine.generate(prompt, sampling_params, request_id=request_id)
    except TypeError:
        # Some older versions name the sampling argument "params".
        try:
            stream = engine.generate(prompt, params=sampling_params, request_id=request_id)
        except TypeError:
            # Last resort: minimal args.
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

    return RunResult(
        model=model_label,
        mode=mode_label,
        prompt_name=prompt_name,
        prompt_len_tokens=prompt_tokens,
        output_tokens=generated_tokens,
        ttft_seconds=ttft,
        end_to_end_seconds=total,
        tokens_per_second=tokens_per_s,
    )


async def bench_model(
    cfg: ModelConfig,
    sampling_params: SamplingParams,
    tp_size: int,
    quantization: Optional[str],
    modes: list[str],
) -> List[RunResult]:
    args = AsyncEngineArgs(
        model=cfg.model,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        quantization=quantization,
    )

    engine = AsyncLLMEngine.from_engine_args(args)
    results: List[RunResult] = []

    try:
        for mode in modes:
            for name, prompt in PROMPTS.items():
                result = await _bench_single_prompt(
                    engine=engine,
                    model_label=cfg.model,
                    mode_label=mode,
                    prompt_name=name,
                    prompt=prompt,
                    sampling_params=sampling_params,
                )
                results.append(result)
    finally:
        # vLLM >=0.4 has async shutdown; older releases expose sync.
        shutdown = getattr(engine, "shutdown", None)
        if shutdown:
            maybe_coro = shutdown()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

    return results


async def main_async(args: argparse.Namespace) -> None:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=[] if args.stop is None else args.stop,
    )

    all_results: List[RunResult] = []
    for cfg in args.models:
        model_results = await bench_model(
            cfg,
            sampling_params,
            tp_size=args.tensor_parallel_size,
            quantization=args.quantization,
            modes=args.modes,
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
        "runs": [asdict(r) for r in all_results],
    }

    # Always write next to this script to avoid path confusion when run from nested dirs.
    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / "benchmarks" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved live benchmark results to {out_path} at {report['generated_at_iso']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live vLLM benchmarks")
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
        help="Tensor parallel degree (set to 2 for 70B on 2×A100, etc.)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Optional vLLM quantization mode (e.g., awq, gptq, fp8) with a compatible checkpoint.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "cachegen"],
        help="Benchmark modes to run (labels only; cachegen hook not yet wired into vLLM).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.models:
        args.models = [
            ModelConfig(model="facebook/opt-125m", max_model_len=2048, dtype="float16"),
            ModelConfig(model="meta-llama/Meta-Llama-3-8B-Instruct", max_model_len=8192, dtype="bfloat16"),
        ]

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
