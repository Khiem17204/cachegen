"""CacheGenEncoder: orchestrates the full KV cache compression pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .chunker import Chunker
from .quantizer import Quantizer
from .delta import DeltaEncoder
from .entropy import EntropyCoder


@dataclass
class EncodedChunk:
    """Container for a single compressed KV cache chunk.

    Attributes
    ----------
    data : bytes
        Zstd-compressed byte stream of the delta-encoded, quantized tensor.
    scales : torch.Tensor
        FP16 scale factors needed by the decoder to dequantize.
        Shape: ``original_shape[:-1] + (1,)`` (before chunking).
    original_dtype : torch.dtype
        Dtype of the tensor *before* quantization (e.g. ``torch.float16``).
    original_shape : tuple[int, ...]
        Shape of the chunk tensor *before* quantization.
    """

    data: bytes
    scales: torch.Tensor
    original_dtype: torch.dtype
    original_shape: tuple[int, ...]


class CacheGenEncoder:
    """End-to-end encoder: chunk → quantize → delta-encode → zstd compress.

    Parameters
    ----------
    chunk_size : int
        Number of tokens per chunk (passed to :class:`Chunker`).
    compression_level : int
        Zstd compression level 1–22 (passed to :class:`EntropyCoder`).
    """

    def __init__(
        self,
        chunk_size: int = 64,
        compression_level: int = 3,
    ) -> None:
        self.chunker = Chunker(chunk_size=chunk_size)
        self.quantizer = Quantizer()
        self.delta_encoder = DeltaEncoder(dim=-2)  # seq_len axis
        self.entropy_coder = EntropyCoder(level=compression_level)

    def encode(self, kv_cache: torch.Tensor) -> list[EncodedChunk]:
        """Compress a full KV cache tensor.

        Parameters
        ----------
        kv_cache : torch.Tensor
            Shape ``[num_layers, 2, num_heads, seq_len, head_dim]``,
            typically FP16.

        Returns
        -------
        list[EncodedChunk]
            One :class:`EncodedChunk` per chunk, containing the compressed
            bytes plus metadata required for decoding.
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
