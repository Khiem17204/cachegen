# Encoder

## Objective

Compress KV cache tensors into chunked payloads using a chunk → quantize → delta-encode → zstd pipeline. The encoder returns `EncodedChunk` objects that contain both compressed bytes and the per-chunk metadata required for decoding.

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
| `scales` | `torch.Tensor` | FP16 dequantization scale factors; required sidecar data for decoding and part of the true encoded size |
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
    total_bytes = len(chunk.data) + chunk.scales.numel() * chunk.scales.element_size()
    print(
        f"compressed={len(chunk.data)} bytes, "
        f"scale_bytes={chunk.scales.numel() * chunk.scales.element_size()}, "
        f"total_payload={total_bytes}"
    )
```

`scales` are not optional metadata: the decoder needs them to reconstruct values, and compression accounting should include both `chunk.data` and the serialized size of `chunk.scales`.

## Source of Truth

If this README and runtime behavior diverge, treat `encoder/encoder.py` and `tests/test_encoder.py` as authoritative.

## Testing

```bash
# Run the encoder test suite
pytest tests/test_encoder.py -v
```
