"""EntropyDecoder: zstd-based byte-stream decompression."""

from __future__ import annotations

import zstandard as zstd


class EntropyDecoder:
    """Decompress byte buffers with *zstandard*.

    Matches the zstd compression used by :class:`EntropyCoder`.
    """

    def __init__(self) -> None:
        self._decompressor = zstd.ZstdDecompressor()

    def decompress(self, data: bytes) -> bytes:
        """Decompress *data* and return the original bytes.
        
        Parameters
        ----------
        data : bytes
            Zstd-compressed payload.
            
        Returns
        -------
        bytes
            The uncompressed byte stream.
        """
        return self._decompressor.decompress(data)
