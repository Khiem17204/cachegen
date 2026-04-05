# Benchmark Results (Stage 5)

## Live vLLM harness
`run_benchmarks.py` now targets a real vLLM server/GPU and writes `benchmarks/results.json`.

Minimal Colab (A100) setup:
```
pip install "vllm>=0.4.0" --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python run_benchmarks.py --model "facebook/opt-125m|2048|float16" --model "mistralai/Mistral-7B-Instruct-v0.3|8192|bfloat16" --max-tokens 64
python visualize_results.py
```

Latest Live vLLM Benchmark (A100, Colab)

| Model | Avg TTFT (ms) | Avg Throughput (tok/s) |
| --- | ---: | ---: |
| facebook/opt-125m | 277.0 | 560.9 |
| mistralai/Mistral-7B-Instruct-v0.3 | 25.4 | 83.3 |

Prompts: short/medium/long suite from Stage 5; max_tokens=64; sequential requests.
