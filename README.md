# cachegen

This repository contains the experimental plan and architecture for reproducing the core claims of **CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving**.

## Primary Goal

Verify that KV cache compression yields a **3–4x compression ratio**, **reduces KV loading latency** over networks, and maintains **minimal impact on model quality**.

---

## Stage 1 — KV Cache Extraction

A standalone, reusable module for intercepting raw FP16 KV tensors from any HuggingFace causal language model.

### Installation

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `transformers`, `pytest`.

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_extraction_hf import KVCacheExtractor

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

extractor = KVCacheExtractor(model, tokenizer)
kv_tensor = extractor.extract("Hello, world!")

print(kv_tensor.shape)   # [12, 2, 12, 4, 64]  for GPT-2
print(kv_tensor.dtype)   # torch.float16
```

### API Reference

#### `KVCacheExtractor(model, tokenizer, device=None)`

| Parameter   | Type                       | Description |
|-------------|----------------------------|-------------|
| `model`     | `PreTrainedModel`          | A loaded HuggingFace causal LM (GPT-2, LLaMA, Mistral, etc.) |
| `tokenizer` | `PreTrainedTokenizerBase`  | The corresponding tokenizer |
| `device`    | `str \| None`              | Target device. Auto-detects `cuda` → `mps` → `cpu` if `None` |

#### `extractor.extract(prompt: str) -> torch.Tensor`

Runs a single prefill forward pass and returns the KV cache reshaped into:

```
[num_layers, 2, num_kv_heads, seq_len, head_dim]
```

- Dimension `1` (size 2) = `[key, value]`
- Output is always **FP16** and **contiguous** in memory
- Supports Grouped Query Attention (GQA) — `num_kv_heads` may differ from `num_attention_heads`

### Using in External Projects

Copy the `kv_extraction_hf/` directory into your project and import:

```python
from kv_extraction_hf import KVCacheExtractor
```

The module depends only on `torch` and `transformers` — no other frameworks required.

### Running Tests

```bash
python -m pytest tests/test_extractor.py -v
```

### Example Script

```bash
python example.py
# → Saves kv_cache.pt to disk and prints shape/size info
```

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
* **Sub-objectives:** * Map the chunking logic to vLLM's existing PagedAttention physical blocks.
* Simulate network bandwidth constraints (e.g., 100 MB/s to 1 GB/s) to stream compressed blocks instead of full raw tensors.



### Stage 5: Evaluation & Baselines

* **Objective:** Establish non-compressed baselines in vLLM and run comparative experiments.
* **Metrics to Capture:**
* *Compression Ratio:* Original size vs. compressed size.
* *Load/Decode Latency:* Ensure decode overhead (`Transfer Time + Decode Time`) is faster than raw transfer time.
* *Time to First Token (TTFT):* Measure end-to-end load and generation speed in the vLLM server.
* *Model Quality:* Verify output accuracy using perplexity, BLEU, and ROUGE on tasks like QA and summarization.


## Project Architecture & Scope

* **Modular Structure:** Code is divided cleanly into `kv_extraction_hf/`, `encoder/`, `decoder/`, `vllm_integration/`, and `experiments/`.
* **MVP Simplification:** Prioritize basic quantization and `zstd` compression over complex custom entropy coding for the initial pass.
