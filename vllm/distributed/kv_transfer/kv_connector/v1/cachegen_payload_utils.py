# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure helpers for CacheGenConnector payload serialization and stats."""

from __future__ import annotations

import io
from typing import Any

import torch


def empty_transfer_stats() -> dict[str, float | int | bool]:
    return {
        "raw_tensor_bytes": 0.0,
        "transmitted_bytes": 0.0,
        "transport_ratio": 1.0,
        "network_time": 0.0,
        "decode_time": 0.0,
        "cachegen_applied": False,
        "raw_fallback_layers": 0,
        "raw_bytes": 0.0,
        "compressed_bytes": 0.0,
        "compression_ratio": 1.0,
    }


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_transfer_stats(
    *,
    raw_tensor_bytes: int,
    transmitted_bytes: int,
    network_time: float,
    decode_time: float,
    cachegen_applied: bool,
    raw_fallback_layers: int,
) -> dict[str, float | int | bool]:
    ratio = (
        float(raw_tensor_bytes) / float(transmitted_bytes)
        if raw_tensor_bytes > 0 and transmitted_bytes > 0
        else 1.0
    )
    return {
        "raw_tensor_bytes": float(raw_tensor_bytes),
        "transmitted_bytes": float(transmitted_bytes),
        "transport_ratio": ratio,
        "network_time": float(network_time),
        "decode_time": float(decode_time),
        "cachegen_applied": bool(cachegen_applied),
        "raw_fallback_layers": int(raw_fallback_layers),
        "raw_bytes": float(raw_tensor_bytes),
        "compressed_bytes": float(transmitted_bytes),
        "compression_ratio": ratio,
    }


def merge_transfer_stats(
    lhs: dict[str, float | int | bool],
    rhs: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    raw_tensor_bytes = _float(lhs.get("raw_tensor_bytes")) + _float(
        rhs.get("raw_tensor_bytes")
    )
    transmitted_bytes = _float(lhs.get("transmitted_bytes")) + _float(
        rhs.get("transmitted_bytes")
    )
    ratio = raw_tensor_bytes / transmitted_bytes if transmitted_bytes > 0 else 1.0
    return {
        "raw_tensor_bytes": raw_tensor_bytes,
        "transmitted_bytes": transmitted_bytes,
        "transport_ratio": ratio,
        "network_time": _float(lhs.get("network_time")) + _float(rhs.get("network_time")),
        "decode_time": _float(lhs.get("decode_time")) + _float(rhs.get("decode_time")),
        "cachegen_applied": _bool(lhs.get("cachegen_applied")) or _bool(
            rhs.get("cachegen_applied")
        ),
        "raw_fallback_layers": _int(lhs.get("raw_fallback_layers")) + _int(
            rhs.get("raw_fallback_layers")
        ),
        "raw_bytes": raw_tensor_bytes,
        "compressed_bytes": transmitted_bytes,
        "compression_ratio": ratio,
    }


def serialize_payload_to_bytes(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def load_payload_from_bytes(blob: bytes) -> dict[str, Any]:
    buffer = io.BytesIO(blob)
    try:
        return torch.load(buffer, map_location="cpu", weights_only=False)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")
