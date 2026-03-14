"""Tests for the CacheGenDecoder and its integration with CacheGenEncoder."""

import torch

from encoder import CacheGenEncoder
from decoder import CacheGenDecoder


def test_decoder_round_trip() -> None:
    """Test full encode → decode pipeline and bounded MSE."""
    # 1. Setup dummy KV cache tensor
    # Shape: [num_layers, 2 (k/v), num_heads, seq_len, head_dim]
    shape = [2, 2, 4, 128, 64]
    
    # We use normally distributed random numbers matching typical normalized activations
    original = torch.randn(*shape, dtype=torch.float16)

    # 2. Encode
    encoder = CacheGenEncoder(chunk_size=64, compression_level=1)
    encoded_chunks = encoder.encode(original)

    # 3. Decode
    decoder = CacheGenDecoder()
    reconstructed = decoder.decode(encoded_chunks)

    # 4. Verify Shape and Dtype
    assert reconstructed.shape == original.shape, "Reconstructed shape does not match original."
    assert reconstructed.dtype == original.dtype, "Reconstructed dtype does not match original."

    # 5. Measure Mean Squared Error (MSE)
    # Cast to float32 for stable MSE calculation
    mse = torch.nn.functional.mse_loss(
        original.to(torch.float32), 
        reconstructed.to(torch.float32)
    )
    
    print(f"\nEncoder -> Decoder Quantization MSE: {mse.item():.6f}")

    # The max-abs symmetric UINT8 compression bounds the absolute quantization error to max / 127
    # An MSE of < 0.05 is typically expected for N(0,1) tensors.
    assert mse.item() < 0.05, f"MSE {mse.item()} exceeded acceptable threshold 0.05"


if __name__ == "__main__":
    test_decoder_round_trip()
