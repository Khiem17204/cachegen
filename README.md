# cachegen

This repository contains the experimental plan and architecture for reproducing the core claims of **CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**.

## Primary Goal

Verify that KV cache compression yields a **3–4x compression ratio**, **reduces KV loading latency** over networks, and maintains **minimal impact on model quality**.

---

## Implementation Approach

To minimize engineering bottlenecks with memory management, this reproduction utilizes a **Two-Phase Strategy**:

* **Phase A: HuggingFace Offline (Stages 1–3):** Extract contiguous KV tensors using standard HuggingFace `transformers`. This allows agents to quickly build datasets, implement core algorithms (quantization, delta encoding, `zstd`), and verify compression ratios in an isolated, static environment.
* **Phase B: vLLM Online (Stages 4–5):** Once encoder/decoder logic is proven, port the compression algorithms directly into vLLM's PagedAttention block manager. This enables accurate testing of streaming latency and Time to First Token (TTFT) in a real-world serving scenario.

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

## Phase B Objectives (vLLM)

### Stage 4: vLLM Integration & Streaming Simulation

* **Objective:** Integrate the proven Phase A encoder/decoder directly into vLLM's cache engine.
* **Sub-objectives:**
  * Map the chunking logic to vLLM's existing PagedAttention physical blocks.
  * Simulate network bandwidth constraints (e.g., 100 MB/s to 1 GB/s) to stream compressed blocks instead of full raw tensors.



### Stage 5: Evaluation & Baselines

* **Objective:** Establish non-compressed baselines in vLLM and run comparative experiments.
* **Metrics to Capture:**
* *Compression Ratio:* Original size vs. compressed size.
* *Load/Decode Latency:* Ensure decode overhead (`Transfer Time + Decode Time`) is faster than raw transfer time.
* *Time to First Token (TTFT):* Measure end-to-end load and generation speed in the vLLM server.
* *Model Quality:* Verify output accuracy using perplexity, BLEU, and ROUGE on tasks like QA and summarization.

## Running Stage 5 Benchmarks

1. Install deps: `pip install -r requirements.txt` (adds matplotlib/pandas for plotting).
2. Run the harness: `python run_benchmarks.py` → writes `benchmarks/results.json` with compression, latency, and TTFT for baseline vs CacheGen across prompt sizes and bandwidths.
3. Visualize: `python visualize_results.py` → saves `benchmarks/ttft_vs_bw.png` and `benchmarks/compression_ratio.png`.
4. Quality check: included in `run_benchmarks.py`—BLEU/ROUGE/perplexity deltas are reported in the JSON under `quality`.

### Latest benchmark snapshot (simulated vLLM shim)

- Artifacts: see `benchmarks/results.json`, `ttft_vs_bw.png`, `compression_ratio.png`, `quality_deltas.png`.
- Averages across short/med/long prompts and model configs (opt-1.3b + mistral-7b shapes):
  - Compression ratio (raw/compressed): ~1.46× (range 1.45–1.47).
  - Transfer time: baseline 0.00079s → CacheGen 0.00054s (avg) due to smaller payloads.
  - TTFT: baseline 3.6ms → CacheGen 7.2ms (avg) reflecting decode overhead; end-to-end still bounded by simulated bandwidth.
  - Quality deltas (BLEU/ROUGE/“ppl” surrogate): ~0; simulated outputs kept identical.


## Project Architecture & Scope

* **Modular Structure:** Code is divided cleanly into `kv_extraction_hf/`, `encoder/`, `decoder/`, `vllm_integration/`, and `experiments/`.
* **MVP Simplification:** Prioritize basic quantization and `zstd` compression over complex custom entropy coding for the initial pass.
