"""CacheGenDecoder: orchestrates the full KV cache decompression pipeline."""

from __future__ import annotations

import torch

from encoder.encoder import EncodedChunk

from .chunker import Dechunker
from .delta import DeltaDecoder
from .entropy import EntropyDecoder
from .quantizer import Dequantizer


class CacheGenDecoder:
    """End-to-end decoder: zstd decompress → delta-decode → dequantize → dechunk.

    Attributes:
        entropy_decoder: ``EntropyDecoder`` for zstd decompression.
        delta_decoder: ``DeltaDecoder`` for cumulative-sum reconstruction.
        dequantizer: ``Dequantizer`` for INT8 → FP16 conversion.
        dechunker: ``Dechunker`` for concatenating chunks.
    """

    def __init__(self) -> None:
        """Initialise the decoder pipeline."""
        self.entropy_decoder = EntropyDecoder()
        self.delta_decoder = DeltaDecoder(dim=-2)  # seq_len axis
        self.dequantizer = Dequantizer()
        self.dechunker = Dechunker()

    def decode(self, encoded_chunks: list[EncodedChunk]) -> torch.Tensor:
        """Decompress a list of ``EncodedChunk`` objects into a full KV cache.

        Args:
            encoded_chunks: Outputs from ``CacheGenEncoder.encode()``.

        Returns:
            Reconstructed tensor of shape
            ``[num_layers, 2, num_heads, seq_len, head_dim]``.
        """
        recovered_chunks: list[torch.Tensor] = []

        for chunk in encoded_chunks:
            # 1. Entropy-decode: zstd compressed bytes → raw tensor bytes
            raw_bytes = self.entropy_decoder.decompress(chunk.data)

            # 2. Reconstruct the 1-D tensor of bytes.  We expect INT16
            #    because DeltaEncoder promotes INT8 to INT16 before saving.
            flat_int16 = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int16)

            # Reshape it to original quantiser shape
            delta_encoded = flat_int16.reshape(chunk.original_shape)

            # 3. Delta-decode: restore original values (casts INT16 → INT8)
            quantized = self.delta_decoder.decode(delta_encoded)

            # 4. Dequantize: INT8 + scales → FP16 (or original float type)
            dequantized = self.dequantizer.dequantize(quantized, chunk.scales)

            # Cast back to the original dtype (e.g. float32 if testing)
            dequantized = dequantized.to(chunk.original_dtype)

            recovered_chunks.append(dequantized)

        # 5. Dechunk: concat all chunks along seq_len axis
        full_tensor = self.dechunker.dechunk(recovered_chunks)

        return full_tensor
