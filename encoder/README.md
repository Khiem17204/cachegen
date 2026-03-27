# Encoder

## Objective

Compress raw FP16 KV cache tensors into compact bitstreams using a chunk → quantize → delta-encode → zstd pipeline, producing `EncodedChunk` objects ready for storage or network transmission in the CacheGen system.

## API Reference

### `CacheGenEncoder`

```python
class CacheGenEncoder:
    def __init__(self, chunk_size: int = 64, compression_level: int = 3) -> None: ...
    def encode(self, kv_cache: torch.Tensor) -> list[EncodedChunk]: ...
```

### `EncodedChunk` (dataclass)

| Field | Type | Description |
|---|---|---|
| `data` | `bytes` | Zstd-compressed bitstream |
| `scales` | `torch.Tensor` | FP16 dequantization scale factors |
| `original_dtype` | `torch.dtype` | Pre-quantization dtype |
| `original_shape` | `tuple[int, ...]` | Pre-quantization chunk shape |

### Component Classes

| Class | Method | Signature |
|---|---|---|
| `Chunker` | `chunk(tensor)` | `torch.Tensor → list[torch.Tensor]` |
| `Quantizer` | `quantize(tensor)` | `torch.Tensor → tuple[torch.Tensor, torch.Tensor]` |
| `DeltaEncoder` | `encode(tensor)` | `torch.Tensor → torch.Tensor` |
| `EntropyCoder` | `compress(data)` / `decompress(data)` | `bytes → bytes` |

## Usage Example

```python
import torch
from encoder import CacheGenEncoder

kv_cache = torch.randn(2, 2, 4, 128, 32, dtype=torch.float16)

enc = CacheGenEncoder(chunk_size=64, compression_level=3)
results = enc.encode(kv_cache)

for chunk in results:
    print(f"compressed={len(chunk.data)} bytes, scales={chunk.scales.shape}")
```

## Testing

```bash
# Run the encoder test suite
pytest tests/test_encoder.py -v
```
