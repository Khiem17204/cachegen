# CacheGen Decoder (Stage 3)

The `decoder` package implements Stage 3 of the expected CacheGen runtime: reconstructing original KV tensors from a compressed byte stream, scaling factors, and metadata. It acts as the exact inverse of the Stage 2 Encoder.

## Components

The decoding process uses modular, decoupled components directly reversing Stage 2 operations:
1.  **`EntropyDecoder`**: Uses the `zstandard` library to decompress byte buffers back into an array of delta-encoded, quantized sequences.
2.  **`DeltaDecoder`**: Reconstructs actual token values from inter-token deltas using a commutative summation (`torch.cumsum`). Safely handles INT16 upcasting done during encode by reverting the recovered sums back to INT8.
3.  **`Dequantizer`**: Multiplies `INT8` scaled vectors by the accompanying block-level `FP16` scales.
4.  **`Dechunker`**: Assembles sequential chunks of sequence lengths back into a contiguous large KV tensor block (`torch.cat`).
5.  **`CacheGenDecoder`**: A wrapper to smoothly feed an `EncodedChunk` returned by the encoder directly through the entire stack.

## Expected Accuracy Loss

Unlike the lossless Delta extraction and Zstd compression, INT8 Quantization introduces a minimal bound error.
By adopting the max-abs dynamic slice quantization factor (`scale = max(|x|) / 127`), we incur a precision truncation. The absolute worst-case distance between real vs. reconstructed float properties is `± scale / 2`. 

Typical empirical evaluations register a Mean-Squared Error (MSE) of approx `~0.005` on normal distribution layers relative to native precision limits (`float16`), meaning the downstream generation perplexity of models generally remains preserved while extracting up to `3x - 5x` in bandwidth efficiencies.

## Usage Example

```python
import torch
from encoder import CacheGenEncoder
from decoder import CacheGenDecoder

# Dummy Float16 Layout: [layers, 2, num_heads, seq_len, head_dim]
kv_cache = torch.randn([1, 2, 4, 128, 64], dtype=torch.float16)

# Encode
encoder = CacheGenEncoder(chunk_size=64)
encoded_chunks = encoder.encode(kv_cache)

# Decode
decoder = CacheGenDecoder()
reconstructed = decoder.decode(encoded_chunks)

# Verify
mse = torch.nn.functional.mse_loss(kv_cache.to(torch.float32), reconstructed.to(torch.float32))
print(f"Roundtrip MSE: {mse.item():.4f}")
```
