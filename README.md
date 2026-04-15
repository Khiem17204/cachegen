# cachegen

This repository contains the experimental plan and architecture for reproducing the core claims of **CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**.

## Primary Goal

Verify that KV cache compression yields a **3–4x compression ratio**, **reduces KV loading latency** over networks, and maintains **minimal impact on model quality**.

---

## Implementation Approach

This repo now uses a true end-to-end path in vLLM:

* **Phase A (offline):** KV extraction + CacheGen encoder/decoder validation.
* **Phase B (online):** source-integrated vLLM KV connector with a runtime toggle:
  * `baseline`: raw transfer (`cachegen_enabled=false`)
  * `cachegen`: compressed transfer (`cachegen_enabled=true`)

The benchmark harness measures TTFT/end-to-end/throughput from real generation
and transfer stats from `RequestOutput.kv_transfer_params`.

---

## Phase A Objectives (HuggingFace)

### Stage 1: Static KV Cache Extraction

* **Objective:** Intercept and extract raw FP16 KV tensors from models like LLaMA or Mistral using HuggingFace.
* **Deliverable:** A baseline dataset of contiguous raw KV tensors saved to disk for offline compression testing.

### Stage 2: Encoder Implementation

* **Objective:** Transform raw KV tensors into a compressed bitstream.
* **Sub-objectives:** * *Chunking:* Split tensors into independent units.
* *Quantization:* Convert FP16 to INT8/INT4 using Max-Abs scaling.
* *Delta Encoding:* Compute differences between neighboring tokens.
* *Entropy Coding:* Apply `zstd` compression to output the final bitstream.



### Stage 3: Decoder Implementation

* **Objective:** Reconstruct KV tensors from the compressed bitstream.
* **Sub-objectives:** Perform inverse operations (entropy decoding, delta reconstruction, dequantization) to verify the tensor can be accurately rebuilt.

---

## Running End-to-End Benchmarks

1. Install deps:
   * `pip install -r requirements.txt`
   * `git submodule update --init --recursive third_party/vllm`
   * `pip install -e ./third_party/vllm`
2. Run the harness:
   * `python run_benchmarks.py --modes baseline cachegen --bandwidth-mbps 100 500 1000`
3. Visualize:
   * `python visualize_results.py`
4. Outputs:
   * `benchmarks/results.json`
   * `benchmarks/ttft_by_model_mode.png`
   * `benchmarks/throughput_by_model_mode.png`
   * `benchmarks/compression_ratio_by_mode.png`
   * `benchmarks/network_time_by_mode_bandwidth.png`

### Latest benchmark snapshot (live vLLM endpoint, `facebook/opt-1.3b`)

- Artifacts: see `benchmarks/live_results_opt13b.json`, `benchmarks/live_results_opt13b.md`, and `benchmarks/live_opt13b_latency.png`.
- Environment: Google Colab Pro, NVIDIA A100 40GB, sequential single-server runs to avoid VRAM contention.
- Real endpoint latency on three prompts:
  - Baseline average: `0.342s`
  - CacheGen average: `0.404s`
  - Delta: CacheGen `+0.063s` on average (`+18.4%`)
- Prompt-level behavior:
  - `short`: `0.485s` → `0.659s`
  - `medium`: `0.275s` → `0.279s`
  - `long`: `0.265s` → `0.275s`
- Findings:
  - On this small model, CacheGen overhead is negligible for medium/long prompts but noticeable on the shortest prompt.
  - Output previews and token counts matched across baseline and CacheGen for all three prompts in this run, so no visible quality drift appeared in the sampled outputs.


## Project Architecture & Scope

* **Modular Structure:** Core code lives in `kv_extraction_hf/`, `encoder/`, `decoder/`, with vLLM source integration under `third_party/vllm/`.
* **MVP Simplification:** Prioritize basic quantization and `zstd` compression over complex custom entropy coding for the initial pass.
