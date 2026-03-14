"""DeltaEncoder: inter-token difference encoding."""

from __future__ import annotations

import torch


class DeltaEncoder:
    """Compute per-token deltas along a configurable axis.

    For a tensor *x* with tokens along ``dim``::

        out[..., 0, :] = x[..., 0, :]
        out[..., t, :] = x[..., t, :] - x[..., t-1, :]   (t ≥ 1)

    When the input dtype is INT8, the output is promoted to **INT16** so that
    the subtraction ``(-128) - (127) = -255`` does not overflow.

    Parameters
    ----------
    dim : int
        Axis along which to compute deltas (default ``-2``, i.e. ``seq_len``).
    """

    def __init__(self, dim: int = -2) -> None:
        self.dim = dim

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute forward deltas.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor (any dtype).

        Returns
        -------
        torch.Tensor
            Same shape as *tensor*.  Dtype is INT16 when the input is INT8,
            otherwise matches the input dtype.
        """
        # Promote INT8 to INT16 to avoid overflow on subtraction.
        if tensor.dtype == torch.int8:
            work = tensor.to(torch.int16)
        else:
            work = tensor

        # torch.diff along self.dim gives shape with dim-size reduced by 1.
        diffs = torch.diff(work, n=1, dim=self.dim)

        # Keep the first element unchanged.
        first = torch.narrow(work, self.dim, 0, 1)

        return torch.cat([first, diffs], dim=self.dim)
