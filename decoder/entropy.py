"""EntropyDecoder: zstd-based byte-stream decompression."""

from __future__ import annotations

import zstandard as zstd


class EntropyDecoder:
    """Decompress byte buffers with *zstandard*.

    Matches the zstd compression used by ``EntropyCoder``.

    Attributes:
        _decompressor: Internal ``ZstdDecompressor`` instance.
    """

    def __init__(self) -> None:
        """Initialise the entropy decoder."""
        self._decompressor = zstd.ZstdDecompressor()

    def decompress(self, data: bytes) -> bytes:
        """Decompress *data* and return the original bytes.

        Args:
            data: Zstd-compressed payload.

        Returns:
            The uncompressed byte stream.
        """
        return self._decompressor.decompress(data)
