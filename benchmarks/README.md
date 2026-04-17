# Benchmarks

This directory contains two categories of files:

- current scripted outputs from the end-to-end vLLM benchmark workflow
- archived/manual artifacts from a separate `facebook/opt-1.3b` live run

## Current Benchmark Workflow

The current benchmark path is:

```bash
pip install -r requirements.txt
git submodule update --init --recursive third_party/vllm
pip install -e ./third_party/vllm
python run_benchmarks.py --modes baseline cachegen --bandwidth-mbps 100 500 1000
python visualize_results.py
```

`run_benchmarks.py` benchmarks two transfer modes through the vendored vLLM connector:

- `baseline`: raw KV transfer with `cachegen_enabled=false`
- `cachegen`: compressed KV transfer with `cachegen_enabled=true`

For each prompt, the script runs:

1. a seed pass that populates external KV storage
2. a measured pass that reloads KV and records transfer-aware metrics

## Current Scripted Outputs

The current scripts produce:

- `results.json`
- `ttft_by_model_mode.png`
- `throughput_by_model_mode.png`
- `compression_ratio_by_mode.png`
- `network_time_by_mode_bandwidth.png`

`results.json` includes the benchmark configuration plus per-run fields such as:

- `ttft_seconds`
- `end_to_end_seconds`
- `tokens_per_second`
- `raw_bytes`
- `compressed_bytes`
- `compression_ratio`
- `network_time`
- `decode_time`
- `bandwidth_mbps`
- `cached_tokens`

## Quality Metrics Caveat

The current end-to-end benchmark workflow does not generate BLEU, ROUGE, perplexity, or `quality_deltas.png`. Do not describe those as current scripted outputs.

## Archived Manual Artifacts

The following files are retained as archived/manual reference artifacts and are not produced by the current benchmark scripts:

- `live_results_opt13b.json`
- `live_results_opt13b.md`
- `live_opt13b_latency.png`

## Repo Boundary

The end-to-end benchmark path depends on vendored vLLM code in `third_party/vllm`. Treat that directory as vendored code unless a task explicitly calls for connector changes there.
