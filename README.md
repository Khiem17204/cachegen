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


## Project Architecture & Scope

* **Modular Structure:** Core code lives in `kv_extraction_hf/`, `encoder/`, `decoder/`, with vLLM source integration under `third_party/vllm/`.
* **MVP Simplification:** Prioritize basic quantization and `zstd` compression over complex custom entropy coding for the initial pass.
