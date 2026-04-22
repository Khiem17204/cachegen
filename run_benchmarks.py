from pathlib import Path
from ttft_benchmark import BenchmarkConfig, ModelConfig, run_benchmark_suite, save_report

config = BenchmarkConfig(
    models=[
        ModelConfig(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_model_len=8192,
            dtype="bfloat16",
        )
    ],
    modes=["quantized_fp8", "cachegen"],
    bandwidth_mbps=[3000.0],
    prompt_lengths=[2048, 4096],
    repeats=5,
    max_tokens=16,
    gpu_memory_utilization=0.85,
    cache_root=Path("/content/cachegen/benchmarks/cache_store"),
    output_path=Path("/content/cachegen/benchmarks/results.json"),
)
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_benchmark_suite(config))