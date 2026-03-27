"""Chunker: splits a KV cache tensor along the sequence-length dimension."""

from __future__ import annotations

import torch


class Chunker:
    """Split a KV cache tensor into fixed-size chunks along ``seq_len``.

    The expected tensor layout is
    ``[num_layers, 2, num_heads, seq_len, head_dim]``
    and chunking is performed along **dim 3** (``seq_len``).

    Attributes:
        chunk_size: Maximum number of tokens per chunk.
    """

    def __init__(self, chunk_size: int = 64) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Maximum number of tokens per chunk.  The last chunk
                may be smaller if ``seq_len`` is not evenly divisible.

        Raises:
            ValueError: If ``chunk_size`` is not positive.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = chunk_size

    def chunk(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Split *tensor* into chunks along the sequence-length axis.

        Args:
            tensor: Input KV cache tensor of shape
                ``[num_layers, 2, num_heads, seq_len, head_dim]``.

        Returns:
            A list of tensors, each with shape
            ``[num_layers, 2, num_heads, ≤chunk_size, head_dim]``.

        Raises:
            ValueError: If *tensor* is not 5-D.
        """
        if tensor.ndim != 5:
            raise ValueError(
                f"Expected 5-D tensor [L, 2, H, S, D], got {tensor.ndim}-D"
            )
        return list(torch.split(tensor, self.chunk_size, dim=3))
