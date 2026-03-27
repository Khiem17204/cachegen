"""CacheGenEncoder: orchestrates the full KV cache compression pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .chunker import Chunker
from .delta import DeltaEncoder
from .entropy import EntropyCoder
from .quantizer import Quantizer


@dataclass
class EncodedChunk:
    """Container for a single compressed KV cache chunk.

    Attributes:
        data: Zstd-compressed byte stream of the delta-encoded,
            quantized tensor.
        scales: FP16 scale factors needed by the decoder to dequantize.
            Shape: ``original_shape[:-1] + (1,)`` (before chunking).
        original_dtype: Dtype of the tensor *before* quantization
            (e.g. ``torch.float16``).
        original_shape: Shape of the chunk tensor *before* quantization.
    """

    data: bytes
    scales: torch.Tensor
    original_dtype: torch.dtype
    original_shape: tuple[int, ...]


class CacheGenEncoder:
    """End-to-end encoder: chunk → quantize → delta-encode → zstd compress.

    Attributes:
        chunker: ``Chunker`` instance for splitting along ``seq_len``.
        quantizer: ``Quantizer`` instance for INT8 quantization.
        delta_encoder: ``DeltaEncoder`` for inter-token differencing.
        entropy_coder: ``EntropyCoder`` for zstd compression.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        compression_level: int = 3,
    ) -> None:
        """Initialise the encoder pipeline.

        Args:
            chunk_size: Number of tokens per chunk (passed to
                ``Chunker``).
            compression_level: Zstd compression level ``1–22`` (passed to
                ``EntropyCoder``).
        """
        self.chunker = Chunker(chunk_size=chunk_size)
        self.quantizer = Quantizer()
        self.delta_encoder = DeltaEncoder(dim=-2)  # seq_len axis
        self.entropy_coder = EntropyCoder(level=compression_level)

    def encode(self, kv_cache: torch.Tensor) -> list[EncodedChunk]:
        """Compress a full KV cache tensor.

        Args:
            kv_cache: Input tensor of shape
                ``[num_layers, 2, num_heads, seq_len, head_dim]``,
                typically FP16.

        Returns:
            A list of ``EncodedChunk`` objects, one per chunk, containing
            the compressed bytes plus metadata required for decoding.
        """
        chunks = self.chunker.chunk(kv_cache)
        encoded_chunks: list[EncodedChunk] = []

        for chunk in chunks:
            original_dtype = chunk.dtype
            original_shape = tuple(chunk.shape)

            # 1. Quantize: FP16 → INT8 + scales
            quantized, scales = self.quantizer.quantize(chunk)

            # 2. Delta-encode along the seq_len axis
            delta = self.delta_encoder.encode(quantized)

            # 3. Entropy-code: tensor bytes → zstd compressed bytes
            raw_bytes = delta.contiguous().numpy().tobytes()
            compressed = self.entropy_coder.compress(raw_bytes)

            encoded_chunks.append(
                EncodedChunk(
                    data=compressed,
                    scales=scales,
                    original_dtype=original_dtype,
                    original_shape=original_shape,
                )
            )

        return encoded_chunks
