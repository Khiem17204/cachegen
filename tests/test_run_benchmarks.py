import pytest

from run_benchmarks import _extract_transfer_stats, parse_model_arg
from ttft_benchmark import (
    RunResult,
    build_claim_check,
    build_prompt_specs,
    get_mode_config,
    validate_cachegen_measured_run,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [index for index, _ in enumerate(text.split(), start=1)]


def _run_result(*, mode: str, ttft_seconds: float, prompt_length: int = 2048) -> RunResult:
    return RunResult(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        mode=mode,
        repetition=1,
        prompt_id=f"len{prompt_length}_sample1",
        prompt_length_tokens_requested=prompt_length,
        prompt_length_tokens=prompt_length,
        prompt_instance_index=0,
        prompt_offset_tokens=0,
        bandwidth_mbps=3000.0,
        output_tokens=16,
        cached_tokens=2048,
        ttft_seconds=ttft_seconds,
        end_to_end_seconds=ttft_seconds + 0.2,
        tokens_per_second=100.0,
        raw_tensor_bytes=1000,
        transmitted_bytes=400,
        transport_ratio=2.5,
        network_time=0.01,
        decode_time=0.02,
        cachegen_enabled=mode == "cachegen",
        cachegen_applied=mode == "cachegen",
        raw_fallback_layers=0,
        kv_cache_dtype="auto" if mode == "cachegen" else "fp8",
        raw_bytes=1000,
        compressed_bytes=400,
        compression_ratio=2.5,
    )


def test_parse_model_arg_defaults() -> None:
    cfg = parse_model_arg("facebook/opt-125m")
    assert cfg.model == "facebook/opt-125m"
    assert cfg.max_model_len == 8192
    assert cfg.dtype == "bfloat16"


def test_mode_mapping_uses_fp8_only_for_quantized_fp8() -> None:
    quantized_fp8 = get_mode_config("quantized_fp8")
    cachegen = get_mode_config("cachegen")
    raw_debug = get_mode_config("raw_debug")

    assert quantized_fp8.kv_cache_dtype == "fp8"
    assert quantized_fp8.cachegen_enabled is False
    assert quantized_fp8.uses_kv_transfer is False
    assert cachegen.kv_cache_dtype == "auto"
    assert cachegen.cachegen_enabled is True
    assert cachegen.uses_kv_transfer is True
    assert raw_debug.kv_cache_dtype == "auto"
    assert raw_debug.cachegen_enabled is False
    assert raw_debug.uses_kv_transfer is False


def test_build_prompt_specs_returns_two_exact_windows_per_length() -> None:
    tokenizer = FakeTokenizer()
    corpus = " ".join(f"token{index}" for index in range(30))

    prompts = build_prompt_specs(
        tokenizer,
        [4, 8],
        instances_per_length=2,
        corpus_text=corpus,
    )

    assert [prompt.prompt_id for prompt in prompts] == [
        "len4_sample1",
        "len4_sample2",
        "len8_sample1",
        "len8_sample2",
    ]
    assert [len(prompt.prompt_token_ids) for prompt in prompts] == [4, 4, 8, 8]
    assert prompts[0].prompt_token_ids != prompts[1].prompt_token_ids
    assert prompts[2].offset_tokens == 0
    assert prompts[3].offset_tokens == 8


def test_extract_transfer_stats_prefers_canonical_fields() -> None:
    stats = _extract_transfer_stats(
        {
            "raw_tensor_bytes": 1024,
            "transmitted_bytes": 256,
            "transport_ratio": 4.0,
            "network_time": 0.25,
            "decode_time": 0.05,
            "cachegen_enabled": True,
            "cachegen_applied": True,
            "raw_fallback_layers": 0,
            "kv_cache_dtype": "auto",
        }
    )
    assert stats["raw_tensor_bytes"] == 1024
    assert stats["transmitted_bytes"] == 256
    assert stats["transport_ratio"] == 4.0
    assert stats["cachegen_applied"] is True
    assert stats["kv_cache_dtype"] == "auto"


def test_extract_transfer_stats_keeps_zero_bytes() -> None:
    stats = _extract_transfer_stats(
        {
            "raw_bytes": 0,
            "compressed_bytes": 0,
        }
    )
    assert stats["raw_bytes"] == 0
    assert stats["compressed_bytes"] == 0
    assert stats["compression_ratio"] == 1.0


def test_validate_measured_run_rejects_cachegen_fallback() -> None:
    with pytest.raises(RuntimeError, match="fell back to raw transfer"):
        validate_cachegen_measured_run(
            cached_tokens=128,
            transfer_stats={
                "cachegen_enabled": True,
                "cachegen_applied": True,
                "raw_fallback_layers": 1,
            },
        )


def test_build_claim_check_classifies_speedup_range() -> None:
    below = build_claim_check(
        [
            _run_result(mode="quantized_fp8", ttft_seconds=1.0),
            _run_result(mode="cachegen", ttft_seconds=0.4),
        ]
    )
    within = build_claim_check(
        [
            _run_result(mode="quantized_fp8", ttft_seconds=1.28),
            _run_result(mode="cachegen", ttft_seconds=0.4),
        ]
    )
    above = build_claim_check(
        [
            _run_result(mode="quantized_fp8", ttft_seconds=2.0),
            _run_result(mode="cachegen", ttft_seconds=0.4),
        ]
    )

    assert below["range_status"] == "below_paper_range"
    assert within["range_status"] == "within_paper_range"
    assert above["range_status"] == "above_paper_range"
