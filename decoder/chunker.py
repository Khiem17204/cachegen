"""Dechunker: reassembles KV cache tensor chunks."""

from __future__ import annotations

import torch


class Dechunker:
    """Concatenates fixed-size chunks along ``seq_len``.

    The expected chunk layout is
    ``[num_layers, 2, num_heads, chunk_size, head_dim]``
    and concatenation is performed along **dim 3** (``seq_len``).
    """

    def __init__(self) -> None:
        # Dim 3 is seq_len in [num_layers, 2 (k/v), num_heads, seq_len, head_dim]
        self.dim = 3

    def dechunk(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate a list of *chunks*.

        Parameters
        ----------
        chunks : list[torch.Tensor]
            Each element has shape
            ``[num_layers, 2, num_heads, l_i, head_dim]``.

        Returns
        -------
        torch.Tensor
            Shape ``[num_layers, 2, num_heads, seq_len, head_dim]``.
        """
        if not chunks:
            raise ValueError("Expected at least one chunk to dechunk.")
            
        return torch.cat(chunks, dim=self.dim)
