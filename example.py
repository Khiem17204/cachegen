#!/usr/bin/env python
"""
Example: Extract KV cache from GPT-2 and compress it using CacheGenEncoder.
This demonstrates the connection between Stage 1 (Extraction) and Stage 2 (Encoding).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_extraction_hf import KVCacheExtractor
from encoder import CacheGenEncoder


def main() -> None:
    # ----- Configuration -----
    model_name = "gpt2"
    prompt = (
        "CacheGen compresses the KV cache of large language models to "
        "reduce streaming latency while preserving generation quality."
    )
    chunk_size = 64
    compression_level = 3

    # ----- Stage 1: KV Cache Extraction -----
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    extractor = KVCacheExtractor(model, tokenizer)
    print("Extracting KV cache from prompt...")
    kv_cache = extractor.extract(prompt)

    raw_bytes = kv_cache.nelement() * kv_cache.element_size()
    print(f"\nExtracted KV shape: {tuple(kv_cache.shape)}")
    print(f"Raw shape logic   : [num_layers, 2 (k/v), num_heads, seq_len, head_dim]")
    print(f"Raw size          : {raw_bytes:,} bytes ({raw_bytes / 1024:.1f} KiB)")

    # ----- Stage 2: Encode -----
    print("\nEncoding via CacheGenEncoder...")
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

    # ----- Stage 3: Decode -----
    print("\nDecoding via CacheGenDecoder...")
    from decoder import CacheGenDecoder
    decoder = CacheGenDecoder()
    reconstructed = decoder.decode(encoded_chunks)

    print(f"\nReconstructed shape: {tuple(reconstructed.shape)}")
    print(f"Reconstructed dtype: {reconstructed.dtype}")
    
    mse = torch.nn.functional.mse_loss(
        kv_cache.to(torch.float32), 
        reconstructed.to(torch.float32)
    )
    print(f"Quantization MSE   : {mse.item():.6f}")


if __name__ == "__main__":
    main()
