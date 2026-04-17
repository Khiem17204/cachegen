# AGENTS.md

## Start Here

1. Create or activate a Python environment for the repo.
2. Install dependencies with `pip install -r requirements.txt`.
3. Read this file, then read `README.md`.
4. Run the baseline tests:
   `pytest tests/test_encoder.py tests/test_decoder.py tests/test_vllm_integration.py -q`
5. If you need the HuggingFace extractor, run:
   `pytest tests/test_extractor.py -q`

## What This Repo Currently Implements

- `kv_extraction_hf/`: extracts contiguous KV tensors from HuggingFace causal language models.
- `encoder/`: chunks KV tensors, quantizes to `int8`, delta-encodes, and compresses with `zstandard`.
- `decoder/`: reverses the serialized pipeline and reconstructs approximate tensors from encoded chunks.
- `vllm_integration/`: a shim that adapts the encoder/decoder to a mock cache-engine workflow for transfer and reinsertion experiments.
- `run_benchmarks.py`: a simulated benchmark harness that compares baseline transfer against the shimmed CacheGen path.
- `visualize_results.py`: plotting utility for benchmark outputs already written to `benchmarks/results.json`.

## Simulated vs Real

Real today:

- HuggingFace KV extraction via `transformers`.
- Encoder and decoder round-trip tests.
- Benchmark file generation for `benchmarks/results.json`, `benchmarks/ttft_vs_bw.png`, and `benchmarks/compression_ratio.png`.

Simulated today:

- `run_benchmarks.py` uses random tensors and a mock vLLM-style cache path rather than a real vLLM cache-engine integration.
- Network transfer is modeled with a software simulator.
- Offline quality metrics in `results.json` are surrogate helpers implemented in the script itself; they are not real paper-grade BLEU, ROUGE, or perplexity evaluation.

Archived/manual artifacts:

- `benchmarks/live_results_opt13b.json`
- `benchmarks/live_results_opt13b.md`
- `benchmarks/live_opt13b_latency.png`

Treat those `opt-1.3b` files as manually captured reference artifacts, not as outputs of the current scripted benchmark pipeline.

## Authoritative Sources

- Source of truth for behavior: code plus tests.
- Source of truth for onboarding: this file and the repo-root `README.md`.
- Source of truth for module usage details: each module README, but if prose and code disagree, trust the code/tests first.

## Validated Setup

- Recommended bootstrap: `pip install -r requirements.txt`
- First-time extractor use needs network access because HuggingFace model/tokenizer weights may need to be downloaded.
- The repo includes a vendored `third_party/` tree; normal cleanup and feature work should stay in the repo-owned modules instead of editing vendored code.

## Validated Commands

- Install dependencies:
  `pip install -r requirements.txt`
- Core tests:
  `pytest tests/test_encoder.py tests/test_decoder.py tests/test_vllm_integration.py -q`
- Extractor test:
  `pytest tests/test_extractor.py -q`
- Generate simulated benchmark data:
  `python run_benchmarks.py`
- Generate current scripted plots:
  `python visualize_results.py`

Current benchmark outputs from those scripts are:

- `benchmarks/results.json`
- `benchmarks/ttft_vs_bw.png`
- `benchmarks/compression_ratio.png`

`visualize_results.py` does not generate `quality_deltas.png`.

## Repo Boundaries

- Do not modify `third_party/` for normal work.
- Do not describe archived benchmark files as current pipeline outputs.
- Do not treat top-level benchmark claims as verified unless the claim can be traced to current code or is explicitly labeled archived/manual.
- Keep cleanup-only changes focused on truthfulness, setup clarity, and reproducible commands.

## Benchmark Caveat

The benchmark flow in this repo is useful for exercising the current encoder/decoder and shim logic, but it is not a full paper reproduction. Offline quality values are partly synthetic today, and the live `opt-1.3b` artifacts are manual archived references rather than files produced by the current automation path.
