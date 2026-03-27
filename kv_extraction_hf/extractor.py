"""KVCacheExtractor: intercept and extract raw FP16 KV tensors.

Extracts KV cache tensors from HuggingFace causal language models.
This module is intentionally decoupled from any training loop or serving
framework.  It depends only on PyTorch and HuggingFace ``transformers``.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class KVCacheExtractor:
    """Extract KV cache tensors from a HuggingFace causal LM.

    The ``extract`` method runs a single prefill forward pass using
    ``use_cache=True`` and reshapes the resulting ``past_key_values``
    into a standardised contiguous tensor of shape
    ``[num_layers, 2, num_kv_heads, seq_len, head_dim]``
    where the ``2`` dimension corresponds to ``[key, value]``.

    Attributes:
        model: The HuggingFace causal language model in eval mode.
        tokenizer: The corresponding tokenizer.
        device: Resolved target device for inference.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialise the extractor.

        Args:
            model: A loaded HuggingFace causal language model (e.g. GPT-2,
                LLaMA, Mistral).  The model will be set to ``eval()`` mode
                automatically.
            tokenizer: The corresponding tokenizer.
            device: Target device.  If ``None``, auto-detected as
                ``cuda`` → ``mps`` → ``cpu`` in that order.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._resolve_device(device)

        # Ensure eval mode and move to target device
        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, prompt: str) -> torch.Tensor:
        """Run a prefill pass and return the KV cache as a contiguous FP16 tensor.

        Args:
            prompt: The text prompt to feed through the model.

        Returns:
            A ``torch.Tensor`` of shape
            ``[num_layers, 2, num_kv_heads, seq_len, head_dim]``,
            dtype ``torch.float16``, contiguous in memory.
        """
        input_ids = self._tokenize(prompt)

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)

        past_key_values = outputs.past_key_values
        return self._reshape(past_key_values)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, prompt: str) -> torch.Tensor:
        """Tokenize *prompt* and return ``input_ids`` on the target device.

        Args:
            prompt: Raw text string to tokenize.

        Returns:
            A 2-D ``torch.Tensor`` of token IDs with shape ``[1, seq_len]``.
        """
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return encoded["input_ids"].to(self.device)

    @staticmethod
    def _reshape(
        past_key_values: Any,
    ) -> torch.Tensor:
        """Reshape HuggingFace ``past_key_values`` into a single tensor.

        Supports both:

        * **Legacy format** (transformers < 5): a tuple of
          ``(key_tensor, value_tensor)`` tuples.
        * **DynamicCache** (transformers ≥ 5): an object with a ``.layers``
          list where each layer exposes ``.keys`` and ``.values`` tensors.

        In both cases the per-layer tensors have shape
        ``[batch, num_kv_heads, seq_len, head_dim]``.

        Args:
            past_key_values: The ``past_key_values`` output from a
                HuggingFace model forward pass.

        Returns:
            A ``torch.Tensor`` of shape
            ``[num_layers, 2, num_kv_heads, seq_len, head_dim]``,
            contiguous, FP16, on CPU.
        """
        per_layer: list[torch.Tensor] = []

        # ── DynamicCache (transformers ≥ 5) ─────────────────────────────
        if hasattr(past_key_values, "layers"):
            for layer in past_key_values.layers:
                key = layer.keys  # [batch, heads, seq, dim]
                value = layer.values
                kv = torch.stack([key.squeeze(0), value.squeeze(0)], dim=0)
                per_layer.append(kv)
        else:
            # ── Legacy tuple format ──────────────────────────────────────
            for key, value in past_key_values:
                kv = torch.stack([key.squeeze(0), value.squeeze(0)], dim=0)
                per_layer.append(kv)

        # Stack across layers → [num_layers, 2, num_kv_heads, seq_len, head_dim]
        tensor = torch.stack(per_layer, dim=0)
        # Move to CPU before FP16 conversion to avoid MPS issues on macOS
        return tensor.cpu().contiguous().half()

    @staticmethod
    def _resolve_device(
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.device:
        """Auto-detect the best available device.

        Args:
            device: Explicit device string or ``torch.device``.  If ``None``,
                the method probes ``cuda`` → ``mps`` → ``cpu``.

        Returns:
            The resolved ``torch.device``.
        """
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
