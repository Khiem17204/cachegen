# Benchmarks

This directory contains two categories of files:

- current scripted outputs from the TTFT-focused vLLM benchmark workflow
- archived/manual artifacts from a separate `facebook/opt-1.3b` live run

## Current Benchmark Workflow

The current benchmark path is:

```bash
pip install -r requirements.txt
git submodule update --init --recursive third_party/vllm
pip install -e ./third_party/vllm
python run_benchmarks.py \
  --modes quantized_fp8 cachegen \
  --bandwidth-mbps 3000 \
  --prompt-lengths 2048 4096 \
  --repeats 5
python visualize_results.py
```

`run_benchmarks.py` benchmarks a plain baseline against the vendored CacheGen connector path:

- `quantized_fp8`: plain generation with `kv_cache_dtype="fp8"`
- `cachegen`: CacheGen transfer with `cachegen_enabled=true`, `kv_cache_dtype="auto"`

For each `(model, mode, bandwidth)` combination, the script:

1. warms the engine once
2. builds deterministic tokenizer-exact prompts from `prompt_corpus.txt`
3. uses the plain one-pass path for `quantized_fp8`
4. clears the cache store for every `cachegen` seed/measured prompt pair
5. runs a seed pass that populates external KV storage for `cachegen`
6. runs a measured pass that reloads KV and records transfer-aware metrics for `cachegen`

## Current Scripted Outputs

The current scripts produce:

- `results.json`
- `ttft_comparison_3gbps.png`
- `transport_breakdown_3gbps.png`

`results.json` includes the benchmark configuration plus per-run fields such as:

- `prompt_length_tokens_requested`
- `repetition`
- `ttft_seconds`
- `end_to_end_seconds`
- `tokens_per_second`
- `raw_tensor_bytes`
- `transmitted_bytes`
- `transport_ratio`
- `network_time`
- `decode_time`
- `cachegen_enabled`
- `cachegen_applied`
- `raw_fallback_layers`
- `kv_cache_dtype`
- `bandwidth_mbps`
- `cached_tokens`

The report schema also includes:

- `meta`
- `config`
- `summaries`
- `claim_check`

## Quality Metrics Caveat

The current end-to-end benchmark workflow does not generate BLEU, ROUGE, perplexity, or `quality_deltas.png`. Do not describe those as current scripted outputs.

## Archived Manual Artifacts

The following files are retained as archived/manual reference artifacts and are not produced by the current benchmark scripts:

- `live_results_opt13b.json`
- `live_results_opt13b.md`
- `live_opt13b_latency.png`

## Repo Boundary

The end-to-end benchmark path depends on vendored vLLM code in `third_party/vllm`. Treat that directory as vendored code unless a task explicitly calls for connector changes there.
