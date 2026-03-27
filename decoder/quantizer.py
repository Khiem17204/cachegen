"""Dequantizer: INT8 → FP16 decompression."""

from __future__ import annotations

import torch


class Dequantizer:
    """Restore floating-point values from INT8 using Max-Abs scales.

    The dequantized value is::

        x_fp = x_q * scale

    Where ``scale = max(|x|) / 127`` was computed during encoding.
    """

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    @staticmethod
    def dequantize(tensor: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Dequantize an INT8 tensor back to floating point.

        Args:
            tensor: INT8 quantized tensor of arbitrary shape.
            scales: FP16 scale factors with shape
                ``tensor.shape[:-1] + (1,)``.

        Returns:
            Reconstructed FP16 tensor with the same shape as *tensor*.
        """
        # Multiply the INT8 tensor by the scale factors.
        # We cast the INT8 tensor to the dtype of the scales (e.g. FP16)
        # before multiplication to maintain precision.
        recovered = tensor.to(scales.dtype) * scales
        return recovered
