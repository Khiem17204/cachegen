#!/usr/bin/env python
"""Example: compress a dummy FP16 KV cache through the CacheGen encoder."""

import torch
from encoder import CacheGenEncoder


def main() -> None:
    # ----- Configuration -----
    num_layers = 2
    num_heads = 8
    seq_len = 256
    head_dim = 64
    chunk_size = 64
    compression_level = 3

    # ----- Create a dummy KV cache -----
    kv_cache = torch.randn(
        num_layers, 2, num_heads, seq_len, head_dim, dtype=torch.float16
    )

    raw_bytes = kv_cache.nelement() * kv_cache.element_size()
    print(f"Input KV shape : {tuple(kv_cache.shape)}")
    print(f"Raw size       : {raw_bytes:,} bytes ({raw_bytes / 1024:.1f} KiB)")

    # ----- Encode -----
    encoder = CacheGenEncoder(
        chunk_size=chunk_size,
        compression_level=compression_level,
    )
    encoded_chunks = encoder.encode(kv_cache)

    total_data = sum(len(c.data) for c in encoded_chunks)
    total_scales = sum(c.scales.nelement() * c.scales.element_size() for c in encoded_chunks)
    total_compressed = total_data + total_scales

    print(f"\nChunks         : {len(encoded_chunks)}")
    print(f"Compressed data: {total_data:,} bytes")
    print(f"Scale overhead : {total_scales:,} bytes")
    print(f"Total encoded  : {total_compressed:,} bytes ({total_compressed / 1024:.1f} KiB)")
    print(f"Compression    : {raw_bytes / total_compressed:.2f}x")


if __name__ == "__main__":
    main()
