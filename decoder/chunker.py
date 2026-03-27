"""Dechunker: reassembles KV cache tensor chunks."""

from __future__ import annotations

import torch


class Dechunker:
    """Concatenates fixed-size chunks along ``seq_len``.

    The expected chunk layout is
    ``[num_layers, 2, num_heads, chunk_size, head_dim]``
    and concatenation is performed along **dim 3** (``seq_len``).

    Attributes:
        dim: Concatenation axis (always ``3``).
    """

    def __init__(self) -> None:
        """Initialise the dechunker."""
        # Dim 3 is seq_len in [num_layers, 2 (k/v), num_heads, seq_len, head_dim]
        self.dim = 3

    def dechunk(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate a list of *chunks* back into a full tensor.

        Args:
            chunks: A list of tensors, each with shape
                ``[num_layers, 2, num_heads, l_i, head_dim]``.

        Returns:
            A single tensor of shape
            ``[num_layers, 2, num_heads, seq_len, head_dim]``.

        Raises:
            ValueError: If *chunks* is empty.
        """
        if not chunks:
            raise ValueError("Expected at least one chunk to dechunk.")

        return torch.cat(chunks, dim=self.dim)
