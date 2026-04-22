"""Utilities for converting HuggingFace KV caches to CacheGen tensors."""

from __future__ import annotations

from typing import Any

import torch


def past_key_values_to_tensor(
    past_key_values: Any,
    *,
    dtype: torch.dtype | None = torch.float16,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert model ``past_key_values`` to ``[layers, 2, heads, seq, dim]``.

    The offline benchmark target is GPT-2, whose cache is tuple-style.  The
    extractor also accepts simple DynamicCache-like objects to preserve its
    existing behavior.
    """
    per_layer: list[torch.Tensor] = []

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                raise TypeError("Dynamic cache layers must expose keys and values")
            per_layer.append(_stack_layer(key, value))
    else:
        for key, value in past_key_values:
            per_layer.append(_stack_layer(key, value))

    if not per_layer:
        raise ValueError("past_key_values must contain at least one layer")

    tensor = torch.stack(per_layer, dim=0).to(device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.cpu().contiguous()


def tensor_to_past_key_values(
    kv_tensor: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    like: Any | None = None,
    model_config: Any | None = None,
) -> Any:
    """Convert a standardized KV tensor back to a HuggingFace cache.

    By default this returns GPT-2 tuple-style caches.  If ``like`` is a
    DynamicCache-style object, the returned cache matches that newer API.
    """
    if kv_tensor.ndim != 5:
        raise ValueError(
            f"Expected 5-D tensor [layers, 2, heads, seq, dim], got {kv_tensor.ndim}-D"
        )
    if kv_tensor.shape[1] != 2:
        raise ValueError(f"Expected dim 1 to contain key/value pair, got {kv_tensor.shape[1]}")

    tensor = kv_tensor
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device, dtype=dtype)

    if like is not None and hasattr(like, "layers"):
        return _tensor_to_dynamic_cache(tensor, model_config=model_config)

    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index in range(tensor.shape[0]):
        key = tensor[layer_index, 0].unsqueeze(0).contiguous()
        value = tensor[layer_index, 1].unsqueeze(0).contiguous()
        layers.append((key, value))
    return tuple(layers)


def _tensor_to_dynamic_cache(kv_tensor: torch.Tensor, *, model_config: Any | None) -> Any:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:  # pragma: no cover - depends on transformers version
        raise RuntimeError("DynamicCache reconstruction requires transformers") from exc

    cache = DynamicCache(config=model_config)
    for layer_index in range(kv_tensor.shape[0]):
        key = kv_tensor[layer_index, 0].unsqueeze(0).contiguous()
        value = kv_tensor[layer_index, 1].unsqueeze(0).contiguous()
        cache.update(key, value, layer_index)
    return cache


def _stack_layer(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected per-layer key/value tensors shaped [batch, heads, seq, dim]")
    if key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("Only batch size 1 KV caches are supported")
    if key.shape != value.shape:
        raise ValueError(f"Key/value shapes must match, got {key.shape} and {value.shape}")
    return torch.stack([key.squeeze(0), value.squeeze(0)], dim=0)
