"""Quantizer: Max-Abs FP16 → INT8 quantization."""

from __future__ import annotations

import torch


class Quantizer:
    """Max-Abs symmetric quantization to INT8.

    For each slice along the last dimension the scale factor is::

        scale = max(|x|) / 127

    The quantized value is::

        x_q = clamp(round(x / scale), -128, 127)

    The returned ``scales`` tensor has the same shape as *x* except the last
    dimension is collapsed to 1, and it is kept in FP16 so the decoder can
    dequantize losslessly (up to rounding).
    """

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    @staticmethod
    def quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a floating-point tensor to INT8.

        Parameters
        ----------
        tensor : torch.Tensor
            Arbitrary-shape floating-point tensor (typically FP16).

        Returns
        -------
        quantized : torch.Tensor
            INT8 tensor with the same shape as *tensor*.
        scales : torch.Tensor
            FP16 scale factors.  Shape is ``tensor.shape[:-1] + (1,)``.
        """
        # Compute per-row (last-dim) max-abs value.
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)  # [..., 1]

        # Avoid division by zero for all-zero rows.
        abs_max = abs_max.clamp(min=1e-8)

        scales = (abs_max / 127.0).to(torch.float16)

        quantized = torch.clamp(
            torch.round(tensor / scales), min=-128, max=127
        ).to(torch.int8)

        return quantized, scales
