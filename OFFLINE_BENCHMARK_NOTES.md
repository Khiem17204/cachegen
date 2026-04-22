# Offline Benchmark Notes

## What Was Added

This repo now has an offline benchmark path for two numbers.

- Compression ratio after CacheGen encode and decode.
- Perplexity delta after using the decoded KV cache.

The entrypoint is:

```bash
python run_offline_benchmarks.py
```

It writes:

```text
benchmarks/offline_subset_results.json
```

This path does not use vLLM. It does not measure TTFT. It is meant to be a smaller offline result that can be reproduced on one machine.

## Benchmark Setup

Default setup:

- Model: `gpt2`
- Dataset: `wikitext-2-raw-v1`
- Split: `test`
- Windows: `64`
- Window length: `192` tokens
- Cached prefix: `127` tokens
- Scored continuation: `64` tokens
- Chunk size: `64`
- Compression level: `3`

Each window uses real GPT-2 KV cache tensors. The code compresses the cached prefix. It serializes the compressed chunks. It deserializes them. It decodes them back to KV tensors. Then it scores the same continuation with the decoded cache.

## What The Code Does

For every 192 token window:

1. Tokens `0..126` are used as the cached prefix.
2. Tokens `127..190` are used as the score input.
3. Tokens `128..191` are used as the score target.

The baseline path does this:

1. Run GPT-2 on the cached prefix.
2. Keep the original HuggingFace KV cache.
3. Score the continuation with that original cache.
4. Record the summed negative log likelihood.

The CacheGen path does this:

1. Convert the original HuggingFace cache into a tensor.
2. Encode the tensor with `CacheGenEncoder`.
3. Serialize the encoded chunks into the `CGEN1` byte format.
4. Deserialize the bytes back into encoded chunks.
5. Decode the chunks with `CacheGenDecoder`.
6. Convert the decoded tensor back into a HuggingFace cache.
7. Score the same continuation with that decoded cache.
8. Record the summed negative log likelihood.

The compression ratio uses the serialized bytes. It does not use an estimate.

```text
compression_ratio = raw_tensor_bytes / cachegen_payload_bytes
```

## What To Record

Keep these aggregate values from `benchmarks/offline_subset_results.json`.

- `compression_ratio`
  - Main compression result.
  - Higher is better.
- `raw_tensor_bytes`
  - Size of the original cached prefix tensors.
- `cachegen_payload_bytes`
  - Size of the serialized CacheGen payload.
- `ppl_baseline`
  - Perplexity when using the original KV cache.
- `ppl_cachegen`
  - Perplexity when using the decoded CacheGen KV cache.
- `ppl_delta_abs`
  - Absolute change in perplexity.
  - Computed as `ppl_cachegen - ppl_baseline`.
- `ppl_delta_pct`
  - Percent change in perplexity.
  - Computed relative to baseline.
- `avg_nll_baseline`
  - Average negative log likelihood for the baseline path.
- `avg_nll_cachegen`
  - Average negative log likelihood for the CacheGen path.
- `total_scored_tokens`
  - Total number of continuation tokens used for scoring.
- `windows`
  - Number of benchmark windows.

Current recorded run:

- Compression ratio: `3.2901537098732203`
- Baseline perplexity: `33.028161118016314`
- CacheGen perplexity: `33.021920652640226`
- Perplexity delta: `-0.006240465376087911`
- Perplexity delta percent: `-0.018894377297571804`
- Windows: `64`
- Scored tokens: `4096`

Useful one line summary:

```text
CacheGen got 3.290x compression on GPT-2 KV caches. Perplexity changed from 33.028 to 33.022 on WikiText-2.
```

## Output Shape

The result JSON has three top-level sections.

- `benchmark`
- `config`
- `aggregates`
- `windows`

Use `aggregates` for tables and slides. Use `windows` if you need per-window data.

Each window row includes:

- token offset
- cached prefix token count
- scored token count
- raw bytes
- serialized CacheGen bytes
- compression ratio
- baseline NLL
- CacheGen NLL
- baseline perplexity
- CacheGen perplexity
- perplexity delta

## Files To Know

- `run_offline_benchmarks.py`: main benchmark script.
- `encoder/wire_format.py`: deterministic CacheGen payload format.
- `kv_extraction_hf/cache_conversion.py`: converts HuggingFace caches to and from CacheGen tensors.
- `kv_extraction_hf/extractor.py`: now supports extraction from exact token IDs.
- `benchmarks/offline_subset_results.json`: recorded benchmark output.

## Why These Files Changed

`run_offline_benchmarks.py` was added because this benchmark should not depend on the broken vLLM submodule.

`encoder/wire_format.py` was added so compression ratio uses a real byte payload. This is better than adding up compressed chunk bytes only. It includes metadata and scale tensors.

`kv_extraction_hf/cache_conversion.py` was added so the benchmark has one place for cache conversion. It supports tuple style caches and current Transformers `DynamicCache`.

`kv_extraction_hf/extractor.py` was updated with `extract_from_input_ids()`. This lets a benchmark use exact token windows instead of re-tokenizing text.

`benchmarks/offline_subset_results.json` is now unignored in `.gitignore`. The rest of the JSON files stay ignored.

## Checks Run

```bash
pytest tests/test_cache_conversion.py tests/test_wire_format.py tests/test_offline_benchmarks.py tests/test_encoder.py tests/test_decoder.py tests/test_run_benchmarks.py tests/test_visualize_results.py -q
```

Result:

```text
37 passed
```

The token ID extractor test also passed.

## Commands To Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the benchmark:

```bash
python run_offline_benchmarks.py
```

Run the focused tests:

```bash
pytest tests/test_cache_conversion.py tests/test_wire_format.py tests/test_offline_benchmarks.py tests/test_encoder.py tests/test_decoder.py tests/test_run_benchmarks.py tests/test_visualize_results.py -q
```

Run the token ID extractor check:

```bash
pytest tests/test_extractor.py::TestKVCacheExtractor::test_extract_from_input_ids_matches_prompt_extraction -q
```

## Known Scope

This benchmark is offline only.

It does not claim the full paper result. It only records compression ratio and perplexity delta for GPT-2 on WikiText-2.

The vLLM submodule is still a separate issue. It was not fixed in this change.
