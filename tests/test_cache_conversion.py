import torch

from kv_extraction_hf.cache_conversion import (
    past_key_values_to_tensor,
    tensor_to_past_key_values,
)


def _make_past_key_values() -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            torch.randn(1, 2, 5, 4, dtype=torch.float32),
            torch.randn(1, 2, 5, 4, dtype=torch.float32),
        )
        for _ in range(3)
    )


def test_past_key_values_tensor_round_trip_preserves_shape_and_dtype() -> None:
    past = _make_past_key_values()

    tensor = past_key_values_to_tensor(past, dtype=None)
    restored = tensor_to_past_key_values(tensor)

    assert tensor.shape == (3, 2, 2, 5, 4)
    assert tensor.dtype == torch.float32
    assert len(restored) == len(past)
    for (original_key, original_value), (new_key, new_value) in zip(past, restored):
        assert new_key.shape == original_key.shape
        assert new_value.shape == original_value.shape
        assert new_key.dtype == original_key.dtype
        assert new_value.dtype == original_value.dtype
        assert torch.equal(new_key, original_key)
        assert torch.equal(new_value, original_value)


def test_past_key_values_to_tensor_can_force_fp16_cpu_output() -> None:
    tensor = past_key_values_to_tensor(_make_past_key_values())

    assert tensor.dtype == torch.float16
    assert tensor.device.type == "cpu"
    assert tensor.is_contiguous()


def test_tensor_to_past_key_values_can_match_dynamic_cache_api() -> None:
    from transformers.cache_utils import DynamicCache

    tensor = past_key_values_to_tensor(_make_past_key_values(), dtype=None)
    dynamic_cache = tensor_to_past_key_values(tensor, like=DynamicCache())

    assert dynamic_cache.get_seq_length() == tensor.shape[3]
    assert hasattr(dynamic_cache, "layers")
