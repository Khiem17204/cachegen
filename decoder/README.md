# Decoder

## Objective

Reconstruct original FP16 KV cache tensors from `EncodedChunk` bitstreams by applying the exact inverse of the Stage 2 encoder pipeline (zstd decompress → delta-decode → dequantize → dechunk) in the CacheGen system.

## API Reference

### `CacheGenDecoder`

```python
class CacheGenDecoder:
    def __init__(self) -> None: ...
    def decode(self, encoded_chunks: list[EncodedChunk]) -> torch.Tensor: ...
```

### Component Classes

| Class | Method | Signature |
|---|---|---|
| `EntropyDecoder` | `decompress(data)` | `bytes → bytes` |
| `DeltaDecoder` | `decode(tensor)` | `torch.Tensor → torch.Tensor` |
| `Dequantizer` | `dequantize(tensor, scales)` | `(torch.Tensor, torch.Tensor) → torch.Tensor` |
| `Dechunker` | `dechunk(chunks)` | `list[torch.Tensor] → torch.Tensor` |

## Usage Example

```python
import torch
from encoder import CacheGenEncoder
from decoder import CacheGenDecoder

kv_cache = torch.randn(1, 2, 4, 128, 64, dtype=torch.float16)

encoded = CacheGenEncoder(chunk_size=64).encode(kv_cache)
reconstructed = CacheGenDecoder().decode(encoded)

mse = torch.nn.functional.mse_loss(
    kv_cache.float(), reconstructed.float()
)
print(f"Round-trip MSE: {mse.item():.6f}")  # typical ~0.005
```

## Testing

```bash
# Run the decoder test suite (includes full encode → decode round-trip)
pytest tests/test_decoder.py -v
```
