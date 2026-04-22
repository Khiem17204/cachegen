import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "vllm"
    / "vllm"
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "cachegen_payload_utils.py"
)


spec = importlib.util.spec_from_file_location("cachegen_payload_utils", MODULE_PATH)
assert spec is not None and spec.loader is not None
cachegen_payload_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cachegen_payload_utils)


def test_serialized_byte_accounting_uses_full_blob_size() -> None:
    payload = {
        "kv_format": "cachegen",
        "raw_tensor_bytes": 1024,
        "chunks": [
            {
                "data": b"tiny",
                "scales": torch.ones(4),
                "original_dtype": torch.float16,
                "original_shape": (2, 2),
            }
        ],
    }

    serialized = cachegen_payload_utils.serialize_payload_to_bytes(payload)

    assert len(serialized) > len(payload["chunks"][0]["data"])


def test_transfer_stats_surface_cachegen_applied_and_fallbacks() -> None:
    first = cachegen_payload_utils.build_transfer_stats(
        raw_tensor_bytes=100,
        transmitted_bytes=40,
        network_time=0.1,
        decode_time=0.05,
        cachegen_applied=False,
        raw_fallback_layers=1,
    )
    second = cachegen_payload_utils.build_transfer_stats(
        raw_tensor_bytes=120,
        transmitted_bytes=30,
        network_time=0.2,
        decode_time=0.03,
        cachegen_applied=True,
        raw_fallback_layers=0,
    )

    merged = cachegen_payload_utils.merge_transfer_stats(first, second)

    assert merged["cachegen_applied"] is True
    assert merged["raw_fallback_layers"] == 1
    assert merged["raw_tensor_bytes"] == 220.0
    assert merged["transmitted_bytes"] == 70.0
