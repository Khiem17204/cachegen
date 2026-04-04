# Benchmark Results (Stage 5)

Artifacts produced by `run_benchmarks.py` on 2026-04-04 using the simulated vLLM shim and CacheGen encoder/decoder:

- `results.json`: Raw metrics for baseline vs CacheGen across prompt sizes and bandwidths.
- `ttft_vs_bw.png`: Plot of TTFT against simulated bandwidth (baseline vs CacheGen).
- `compression_ratio.png`: Plot of CacheGen compression ratio across prompts and bandwidths.

Command used:
```
.venv/bin/python run_benchmarks.py
.venv/bin/python visualize_results.py
```
