import asyncio
import os
from pathlib import Path

import pytest

from ttft_benchmark import BenchmarkConfig, ModelConfig, run_benchmark_suite


@pytest.mark.skipif(
    os.environ.get("CACHEGEN_RUN_VLLM_INTEGRATION") != "1",
    reason="Set CACHEGEN_RUN_VLLM_INTEGRATION=1 to run the slow vLLM integration test.",
)
def test_small_model_ttft_benchmark(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        models=[ModelConfig(model="facebook/opt-125m", max_model_len=2048, dtype="float16")],
        modes=["quantized_fp8", "cachegen"],
        bandwidth_mbps=[3000.0],
        prompt_lengths=[1024],
        repeats=1,
        max_tokens=4,
        cache_root=tmp_path / "cache",
        output_path=tmp_path / "results.json",
    )

    report = asyncio.run(run_benchmark_suite(config))
    runs = report["runs"]

    assert runs
    assert all(run["cached_tokens"] > 0 for run in runs)
    cachegen_runs = [run for run in runs if run["mode"] == "cachegen"]
    assert cachegen_runs
    assert all(run["cachegen_applied"] is True for run in cachegen_runs)
    transmitted_by_mode = {
        run["mode"]: run["transmitted_bytes"]
        for run in runs
    }
    assert transmitted_by_mode["quantized_fp8"] != transmitted_by_mode["cachegen"]
