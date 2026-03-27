"""DeltaDecoder: inter-token difference decoding."""

from __future__ import annotations

import torch


class DeltaDecoder:
    """Reconstruct original values from per-token deltas.

    For a sequence of deltas ``d``::

        out[..., t, :] = sum_{i=0}^t d[..., i, :]

    This mirrors ``DeltaEncoder``.  When decoding deltas that were formed
    from INT8 values (and thus transmitted as INT16), this class safely
    accumulates and casts back to INT8.

    Attributes:
        dim: Axis along which to compute the cumulative sum.
    """

    def __init__(self, dim: int = -2) -> None:
        """Initialise the delta decoder.

        Args:
            dim: Axis along which to compute the cumulative sum
                (default ``-2``, i.e. ``seq_len``).
        """
        self.dim = dim

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute the cumulative sum to invert delta encoding.

        Args:
            tensor: Delta-encoded tensor (e.g. INT16).

        Returns:
            The reconstructed tensor.  If the input was INT16, it is cast
            back to INT8 assuming it originated from INT8 quantized values.
        """
        # Compute prefix sums along the sequence length dimension.
        # This exactly inverts the x[t] - x[t-1] operation.
        recovered = torch.cumsum(tensor, dim=self.dim)

        # If the delta stream was INT16 (promoted during encoding), cast
        # back to INT8 since we know the original quantized values were INT8.
        if tensor.dtype == torch.int16:
            recovered = recovered.to(torch.int8)

        return recovered
