"""Tests for the CacheGen Stage 2 encoder pipeline."""

import torch
import pytest

from encoder import Chunker, Quantizer, DeltaEncoder, EntropyCoder, CacheGenEncoder


# ===================================================================== #
#  Helpers                                                               #
# ===================================================================== #

def _make_kv(num_layers=2, num_heads=4, seq_len=128, head_dim=32):
    """Return a random FP16 KV cache tensor."""
    return torch.randn(num_layers, 2, num_heads, seq_len, head_dim, dtype=torch.float16)


# ===================================================================== #
#  Chunker                                                               #
# ===================================================================== #

class TestChunker:
    def test_exact_split(self):
        """seq_len evenly divisible by chunk_size."""
        kv = _make_kv(seq_len=128)
        chunks = Chunker(chunk_size=64).chunk(kv)
        assert len(chunks) == 2
        for c in chunks:
            assert c.shape == (2, 2, 4, 64, 32)

    def test_remainder_split(self):
        """seq_len NOT evenly divisible — last chunk is smaller."""
        kv = _make_kv(seq_len=100)
        chunks = Chunker(chunk_size=64).chunk(kv)
        assert len(chunks) == 2
        assert chunks[0].shape[3] == 64
        assert chunks[1].shape[3] == 36

    def test_chunk_size_larger_than_seq(self):
        """chunk_size > seq_len should produce a single chunk."""
        kv = _make_kv(seq_len=30)
        chunks = Chunker(chunk_size=64).chunk(kv)
        assert len(chunks) == 1
        assert chunks[0].shape[3] == 30

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=0)

    def test_wrong_ndim(self):
        with pytest.raises(ValueError):
            Chunker().chunk(torch.randn(4, 4))


# ===================================================================== #
#  Quantizer                                                             #
# ===================================================================== #

class TestQuantizer:
    def test_output_dtype_and_range(self):
        x = torch.randn(4, 8, dtype=torch.float16)
        q, scales = Quantizer.quantize(x)
        assert q.dtype == torch.int8
        assert scales.dtype == torch.float16
        assert q.min() >= -128
        assert q.max() <= 127

    def test_scales_shape(self):
        x = torch.randn(2, 2, 4, 64, 32, dtype=torch.float16)
        _, scales = Quantizer.quantize(x)
        assert scales.shape == (2, 2, 4, 64, 1)

    def test_roundtrip_error(self):
        """Dequantized values should be within 1 scale unit of the original."""
        x = torch.randn(8, 16, dtype=torch.float16)
        q, scales = Quantizer.quantize(x)
        x_hat = q.to(torch.float16) * scales
        max_err = (x - x_hat).abs().max().item()
        # Error per element ≤ scale (half quantization bin).
        max_scale = scales.max().item()
        assert max_err <= max_scale + 1e-4

    def test_zero_tensor(self):
        """All-zero input should not cause NaN/Inf."""
        x = torch.zeros(4, 8, dtype=torch.float16)
        q, scales = Quantizer.quantize(x)
        assert not torch.isnan(q.float()).any()
        assert not torch.isnan(scales).any()


# ===================================================================== #
#  DeltaEncoder                                                          #
# ===================================================================== #

class TestDeltaEncoder:
    def test_math(self):
        """Manual verification of delta values."""
        x = torch.tensor([[1, 2], [4, 7], [10, 12]], dtype=torch.int8)
        enc = DeltaEncoder(dim=0)
        delta = enc.encode(x)
        # First row unchanged
        assert (delta[0] == x[0].to(torch.int16)).all()
        # Second row: [4-1, 7-2] = [3, 5]
        assert (delta[1] == torch.tensor([3, 5], dtype=torch.int16)).all()
        # Third row: [10-4, 12-7] = [6, 5]
        assert (delta[2] == torch.tensor([6, 5], dtype=torch.int16)).all()

    def test_shape_preserved(self):
        x = torch.randn(2, 2, 4, 64, 32).to(torch.int8)
        delta = DeltaEncoder(dim=-2).encode(x)
        assert delta.shape == x.shape

    def test_int8_promotion(self):
        """Output should be INT16 when input is INT8 (to prevent overflow)."""
        x = torch.randint(-128, 127, (4, 8), dtype=torch.int8)
        delta = DeltaEncoder(dim=0).encode(x)
        assert delta.dtype == torch.int16

    def test_inverse_via_cumsum(self):
        """cumsum of deltas should reconstruct the original tensor."""
        x = torch.randint(-50, 50, (10, 4), dtype=torch.int8)
        delta = DeltaEncoder(dim=0).encode(x)
        reconstructed = torch.cumsum(delta, dim=0)
        assert (reconstructed == x.to(torch.int16)).all()


# ===================================================================== #
#  EntropyCoder                                                          #
# ===================================================================== #

class TestEntropyCoder:
    def test_roundtrip(self):
        ec = EntropyCoder(level=3)
        original = b"hello world " * 100
        compressed = ec.compress(original)
        decompressed = ec.decompress(compressed)
        assert decompressed == original

    def test_compression_reduces_size(self):
        ec = EntropyCoder()
        data = bytes(range(256)) * 100
        compressed = ec.compress(data)
        assert len(compressed) < len(data)

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            EntropyCoder(level=0)
        with pytest.raises(ValueError):
            EntropyCoder(level=23)


# ===================================================================== #
#  CacheGenEncoder (full pipeline)                                       #
# ===================================================================== #

class TestCacheGenEncoder:
    def test_pipeline_runs(self):
        kv = _make_kv(seq_len=128)
        enc = CacheGenEncoder(chunk_size=64, compression_level=3)
        results = enc.encode(kv)
        assert len(results) == 2
        for r in results:
            assert isinstance(r.data, bytes)
            assert len(r.data) > 0
            assert r.scales.dtype == torch.float16
            assert r.original_dtype == torch.float16

    def test_compressed_smaller_than_raw(self):
        kv = _make_kv(seq_len=256)
        enc = CacheGenEncoder(chunk_size=128)
        results = enc.encode(kv)
        raw_bytes = kv.nelement() * kv.element_size()
        total_compressed = sum(len(r.data) for r in results)
        assert total_compressed < raw_bytes
