# cachegen

This repository contains a CacheGen-style prototype with three main pieces:

- HuggingFace KV extraction in `kv_extraction_hf/`
- local encoder/decoder modules in `encoder/` and `decoder/`
- a Colab-ready TTFT benchmark path that uses a vendored vLLM integration under `third_party/vllm/`

The top-level docs in this repo are intended to describe the code as it exists today, not just the original paper-reproduction plan.

## Current State

The repo currently includes:

- offline KV extraction for HuggingFace causal language models
- an `int8`-based serialized compression pipeline with required scale sidecar data
- a shared TTFT benchmark core in `ttft_benchmark.py` used by both the CLI and the Colab notebook
- a `run_benchmarks.py` CLI that compares a plain `quantized_fp8` baseline against connector-backed `cachegen`
- a plotting script in `visualize_results.py` that renders the TTFT and transport plots from `benchmarks/results.json`
- a first-class Colab notebook in `notebooks/ttft_repro_colab.ipynb`

The repo does not currently include:

- a verified reproduction of the paper's headline `3-4x` compression claim across published settings
- real BLEU, ROUGE, or perplexity evaluation in the current benchmark harness
- generated support for `benchmarks/quality_deltas.png`
- a guarantee that archived live benchmark artifacts are reproducible from the current scripts

## What Works

- You can extract KV tensors with `kv_extraction_hf/`.
- You can encode and decode KV tensors with the local encoder/decoder modules.
- You can run a true end-to-end TTFT benchmark where `quantized_fp8` uses plain generation and `cachegen` uses vendored vLLM connector transfer.
- You can render the current TTFT-focused plot set from `benchmarks/results.json` with `visualize_results.py`.
- You can execute the same benchmark code path on Colab with `notebooks/ttft_repro_colab.ipynb`.

## Archived Artifacts

The following files are kept as archived/manual reference material rather than current benchmark outputs:

- `benchmarks/live_results_opt13b.json`
- `benchmarks/live_results_opt13b.md`
- `benchmarks/live_opt13b_latency.png`

## Validated Commands

Bootstrap the repo:

```bash
pip install -r requirements.txt
git submodule update --init --recursive third_party/vllm
pip install -e ./third_party/vllm
```

Run the primary TTFT benchmark harness:

```bash
python run_benchmarks.py \
  --modes quantized_fp8 cachegen \
  --bandwidth-mbps 3000 \
  --prompt-lengths 2048 4096 \
  --repeats 5
```

Current scripted output:

- `benchmarks/results.json`

Render the current plots:

```bash
python visualize_results.py
```

Current scripted plot outputs:

- `benchmarks/ttft_comparison_3gbps.png`
- `benchmarks/transport_breakdown_3gbps.png`

## Benchmark Notes

- If no `--model` flags are provided, `run_benchmarks.py` defaults to `mistralai/Mistral-7B-Instruct-v0.3|8192|bfloat16`.
- The CLI defaults to `--modes quantized_fp8 cachegen --bandwidth-mbps 3000 --prompt-lengths 2048 4096 --repeats 5 --max-tokens 16`.
- Prompt construction is tokenizer-exact and deterministic. The benchmark builds two prompt instances per requested prompt length from `benchmarks/prompt_corpus.txt`.
- The report schema is TTFT-focused and writes `meta`, `config`, `runs`, `summaries`, and `claim_check`.
- `claim_check` explicitly reports whether the observed 3 Gbps speedup is below, within, or above the paper's `3.2-3.7x` range.
- `benchmarks/quality_deltas.png` is not a current pipeline output and should not be referenced as one.

## Repository Boundaries

- `third_party/` vendors external code. Normal repo cleanup should avoid changing vendored files unless the task explicitly requires it.
- When prose and implementation disagree, prefer the code and tests.
