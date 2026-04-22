import torch

from decoder import CacheGenDecoder
from encoder import CacheGenEncoder
from encoder.wire_format import deserialize_encoded_chunks, serialize_encoded_chunks


def test_wire_format_round_trip_preserves_encoded_chunk_fields() -> None:
    kv_tensor = torch.randn(2, 2, 3, 16, 8, dtype=torch.float16)
    encoded = CacheGenEncoder(chunk_size=8, compression_level=3).encode(kv_tensor)

    payload = serialize_encoded_chunks(encoded)
    restored = deserialize_encoded_chunks(payload)

    assert len(payload) > 0
    assert len(restored) == len(encoded)
    for original, new in zip(encoded, restored):
        assert new.data == original.data
        assert torch.equal(new.scales, original.scales)
        assert new.original_dtype == original.original_dtype
        assert new.original_shape == original.original_shape


def test_wire_format_payload_length_is_stable_and_positive() -> None:
    kv_tensor = torch.ones(1, 2, 2, 16, 8, dtype=torch.float16)
    encoded = CacheGenEncoder(chunk_size=16, compression_level=3).encode(kv_tensor)

    first = serialize_encoded_chunks(encoded)
    second = serialize_encoded_chunks(encoded)

    assert len(first) > 0
    assert len(first) == len(second)
    assert first == second


def test_encode_serialize_deserialize_decode_returns_bounded_mse() -> None:
    kv_tensor = torch.randn(1, 2, 2, 16, 8, dtype=torch.float16)
    encoded = CacheGenEncoder(chunk_size=8, compression_level=3).encode(kv_tensor)
    payload = serialize_encoded_chunks(encoded)

    decoded = CacheGenDecoder().decode(deserialize_encoded_chunks(payload))
    mse = torch.nn.functional.mse_loss(kv_tensor.float(), decoded.float())

    assert decoded.shape == kv_tensor.shape
    assert decoded.dtype == kv_tensor.dtype
    assert mse.item() < 0.05
