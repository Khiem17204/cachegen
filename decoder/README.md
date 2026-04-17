# Decoder

## Objective

Reconstruct KV cache tensors from `EncodedChunk` payloads by inverting the serialized pipeline stages (zstd decompress → delta-decode → dequantize → dechunk). The serialization steps are inverted exactly, but the overall round trip is still lossy because the encoder quantizes values before compression.

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
from encoder import CacheGenEncoder, EncodedChunk
from decoder import CacheGenDecoder

kv_cache = torch.randn(1, 2, 4, 128, 64, dtype=torch.float16)

encoded = CacheGenEncoder(chunk_size=64).encode(kv_cache)
reconstructed = CacheGenDecoder().decode(encoded)

mse = torch.nn.functional.mse_loss(
    kv_cache.float(), reconstructed.float()
)
assert isinstance(encoded[0], EncodedChunk)
print(f"Round-trip MSE: {mse.item():.6f}")
```

`CacheGenDecoder.decode()` expects the full `EncodedChunk` objects, including `scales`. Those scale tensors are required sidecar data, so any size or transfer accounting should include them in addition to `chunk.data`.

## Source of Truth

If this README and runtime behavior diverge, treat `decoder/decoder.py` and `tests/test_decoder.py` as authoritative.

## Testing

```bash
# Run the decoder test suite (includes full encode → decode round-trip)
pytest tests/test_decoder.py -v
```
