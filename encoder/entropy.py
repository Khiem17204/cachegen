"""EntropyCoder: zstd-based byte-stream compression."""

from __future__ import annotations

import zstandard as zstd


class EntropyCoder:
    """Compress / decompress byte buffers with *zstandard*.

    Parameters
    ----------
    level : int
        Zstd compression level (1–22, default 3).
    """

    def __init__(self, level: int = 3) -> None:
        if not 1 <= level <= 22:
            raise ValueError(f"level must be in [1, 22], got {level}")
        self._compressor = zstd.ZstdCompressor(level=level)
        self._decompressor = zstd.ZstdDecompressor()

    def compress(self, data: bytes) -> bytes:
        """Compress *data* and return the compressed bytes."""
        return self._compressor.compress(data)

    def decompress(self, data: bytes) -> bytes:
        """Decompress *data* and return the original bytes."""
        return self._decompressor.decompress(data)
