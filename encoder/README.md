# CacheGen Stage 2: Encoder

Modular pipeline for compressing raw FP16 KV cache tensors into compact bitstreams.

**Data flow:** `Raw FP16 → Chunk → Quantize (INT8) → Delta-Encode → zstd Compress`

---

## Components

### 1. `Chunker`

Splits a 5-D KV cache tensor along the **sequence-length** dimension.

| | Description |
|---|---|
| **Input** | `torch.Tensor` of shape `[num_layers, 2, num_heads, seq_len, head_dim]` |
| **Output** | `list[torch.Tensor]` — each chunk has shape `[num_layers, 2, num_heads, ≤chunk_size, head_dim]` |
| **Parameter** | `chunk_size: int` (default 64) |

```python
from encoder import Chunker
chunks = Chunker(chunk_size=64).chunk(kv_tensor)
```

---

### 2. `Quantizer`

Symmetric **Max-Abs INT8** quantization computed per element along the last dimension.

| | Description |
|---|---|
| **Input** | Any floating-point `torch.Tensor` |
| **Output** | Tuple `(quantized: torch.int8, scales: torch.float16)` |
| **Math** | `scale = max(\|x\|) / 127` per row; `x_q = clamp(round(x / scale), -128, 127)` |

Scales shape: `input.shape[:-1] + (1,)`.

```python
from encoder import Quantizer
q, scales = Quantizer.quantize(fp16_tensor)
```

**Dequantization** (for the decoder): `x_hat = q.float() * scales`

---

### 3. `DeltaEncoder`

Computes inter-token differences along a configurable axis.

| | Description |
|---|---|
| **Input** | `torch.Tensor` (typically INT8) and `dim: int` (default `-2`) |
| **Output** | Same-shape tensor of deltas (INT16 if input is INT8) |
| **Math** | `out[0] = x[0]; out[t] = x[t] - x[t-1]` |

```python
from encoder import DeltaEncoder
delta = DeltaEncoder(dim=-2).encode(int8_tensor)
```

**Inverse** (for the decoder): `torch.cumsum(delta, dim=-2)`

---

### 4. `EntropyCoder`

Wraps **zstandard** for byte-level compression.

| | Description |
|---|---|
| **Input** | `bytes` |
| **Output** | Compressed `bytes` |
| **Parameter** | `level: int` (1–22, default 3) |

```python
from encoder import EntropyCoder
ec = EntropyCoder(level=3)
compressed = ec.compress(raw_bytes)
original  = ec.decompress(compressed)
```

---

### 5. `CacheGenEncoder` (Orchestrator)

Chains all components into a single `encode()` call.

| | Description |
|---|---|
| **Input** | `torch.Tensor` `[num_layers, 2, num_heads, seq_len, head_dim]` (FP16) |
| **Output** | `list[EncodedChunk]` |
| **Parameters** | `chunk_size`, `compression_level` |

Each `EncodedChunk` is a dataclass containing:
- `data: bytes` — compressed bitstream
- `scales: torch.Tensor` — FP16 dequantization scales
- `original_dtype: torch.dtype`
- `original_shape: tuple[int, ...]`

```python
from encoder import CacheGenEncoder

enc = CacheGenEncoder(chunk_size=64, compression_level=3)
results = enc.encode(kv_cache)

for chunk in results:
    print(len(chunk.data), chunk.scales.shape)
```

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/test_encoder.py -v

# Run example
python example.py
```

---

## Integration with vLLM PagedAttention

The `EncodedChunk` dataclass is designed to map directly onto vLLM's physical block layout:

1. Set `chunk_size` equal to vLLM's `block_size` (typically 16 or 32 tokens).
2. Each `EncodedChunk` corresponds to one physical block's worth of KV data.
3. The decoder (Stage 3) will reconstruct the full FP16 chunk from `data` + `scales` and write it back into the paged KV buffer.
