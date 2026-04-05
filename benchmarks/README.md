# Benchmark Results (Stage 5)

## Live vLLM harness
`run_benchmarks.py` now targets a real vLLM server/GPU and writes `benchmarks/results.json`.

Minimal Colab (A100) setup:
```
pip install "vllm>=0.4.0"  # pick the CUDA build matching the Colab image
python run_benchmarks.py
python visualize_results.py
```

Examples:
- Default small + 8B models: `python run_benchmarks.py`
- Custom: `python run_benchmarks.py --model meta-llama/Meta-Llama-3-8B-Instruct|8192|bfloat16`
- 70B with 2×A100: `python run_benchmarks.py --model meta-llama/Llama-2-70b-chat-hf|4096|bfloat16 --tensor-parallel-size 2`
- Try 4/8-bit to squeeze onto a single GPU: add `--quantization awq` (requires a compatible quantized checkpoint)
