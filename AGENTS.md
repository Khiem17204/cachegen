# AGENTS.md

## Start Here

1. Create or activate a Python environment for the repo.
2. Install bootstrap dependencies with `pip install -r requirements.txt`.
3. Sync vendored vLLM with `git submodule update --init --recursive third_party/vllm`.
4. Install the vendored vLLM package with `pip install -e ./third_party/vllm`.
5. Read this file, then read `README.md`.
6. Run the baseline tests:
   `pytest tests/test_encoder.py tests/test_decoder.py tests/test_run_benchmarks.py tests/test_visualize_results.py -q`
7. If you need the HuggingFace extractor, run:
   `pytest tests/test_extractor.py -q`

## What This Repo Currently Implements

- `kv_extraction_hf/`: extracts contiguous KV tensors from HuggingFace causal language models.
- `encoder/`: chunks KV tensors, quantizes to `int8`, delta-encodes, and compresses with `zstandard`.
- `decoder/`: reverses the serialized pipeline and reconstructs approximate tensors from encoded chunks.
- `third_party/vllm/`: vendored vLLM source used by the current end-to-end benchmark path.
- `run_benchmarks.py`: a true end-to-end benchmark harness that runs baseline vs CacheGen transfer through vendored vLLM.
- `visualize_results.py`: plotting utility for benchmark outputs written to `benchmarks/results.json`.

## Real vs Limited

Real today:

- HuggingFace KV extraction via `transformers`.
- Encoder and decoder round-trip tests.
- End-to-end vLLM benchmark execution through `AsyncLLMEngine`.
- Transfer metrics collected from `RequestOutput.kv_transfer_params`.
- Plot generation for:
  `benchmarks/ttft_by_model_mode.png`
  `benchmarks/throughput_by_model_mode.png`
  `benchmarks/compression_ratio_by_mode.png`
  `benchmarks/network_time_by_mode_bandwidth.png`

Limited today:

- The current benchmark flow does not generate BLEU, ROUGE, perplexity, or `quality_deltas.png`.
- The repo should not claim a verified reproduction of the paper's full headline results unless those claims are backed by current code and reproducible runs.

Archived/manual artifacts:

- `benchmarks/live_results_opt13b.json`
- `benchmarks/live_results_opt13b.md`
- `benchmarks/live_opt13b_latency.png`

Treat those `opt-1.3b` files as archived reference material rather than outputs of the current scripted benchmark pipeline.

## Authoritative Sources

- Source of truth for behavior: code plus tests.
- Source of truth for onboarding: this file and the repo-root `README.md`.
- Source of truth for module usage details: each module README, but if prose and code disagree, trust the code/tests first.

## Validated Setup

- Recommended bootstrap:
  `pip install -r requirements.txt`
  `git submodule update --init --recursive third_party/vllm`
  `pip install -e ./third_party/vllm`
- First-time extractor use needs network access because HuggingFace model/tokenizer weights may need to be downloaded.
- `third_party/` is vendored code. Normal cleanup and feature work should stay in repo-owned modules unless the task explicitly calls for vendored changes.

## Validated Commands

- Core tests:
  `pytest tests/test_encoder.py tests/test_decoder.py tests/test_run_benchmarks.py tests/test_visualize_results.py -q`
- Extractor test:
  `pytest tests/test_extractor.py -q`
- Generate end-to-end benchmark data:
  `python run_benchmarks.py --modes baseline cachegen --bandwidth-mbps 100 500 1000`
- Generate current scripted plots:
  `python visualize_results.py`

Current benchmark outputs from those scripts are:

- `benchmarks/results.json`
- `benchmarks/ttft_by_model_mode.png`
- `benchmarks/throughput_by_model_mode.png`
- `benchmarks/compression_ratio_by_mode.png`
- `benchmarks/network_time_by_mode_bandwidth.png`

## Repo Boundaries

- Do not modify `third_party/` for normal work unless the task explicitly requires vendored vLLM changes.
- Do not describe archived benchmark files as current pipeline outputs.
- Do not treat top-level benchmark claims as verified unless the claim can be traced to current code or is explicitly labeled archived/manual.
- Keep cleanup-only changes focused on truthfulness, setup clarity, and reproducible commands.
