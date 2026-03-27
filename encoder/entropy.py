"""EntropyCoder: zstd-based byte-stream compression."""

from __future__ import annotations

import zstandard as zstd


class EntropyCoder:
    """Compress and decompress byte buffers with *zstandard*.

    Attributes:
        _compressor: Internal ``ZstdCompressor`` instance.
        _decompressor: Internal ``ZstdDecompressor`` instance.
    """

    def __init__(self, level: int = 3) -> None:
        """Initialise the entropy coder.

        Args:
            level: Zstd compression level in the range ``[1, 22]``
                (default ``3``).

        Raises:
            ValueError: If *level* is outside the valid range.
        """
        if not 1 <= level <= 22:
            raise ValueError(f"level must be in [1, 22], got {level}")
        self._compressor = zstd.ZstdCompressor(level=level)
        self._decompressor = zstd.ZstdDecompressor()

    def compress(self, data: bytes) -> bytes:
        """Compress *data* and return the compressed bytes.

        Args:
            data: Raw byte buffer to compress.

        Returns:
            The zstd-compressed byte string.
        """
        return self._compressor.compress(data)

    def decompress(self, data: bytes) -> bytes:
        """Decompress *data* and return the original bytes.

        Args:
            data: Zstd-compressed byte buffer.

        Returns:
            The decompressed byte string.
        """
        return self._decompressor.decompress(data)
