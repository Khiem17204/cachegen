from run_benchmarks import _extract_transfer_stats, parse_model_arg


def test_parse_model_arg_defaults() -> None:
    cfg = parse_model_arg("facebook/opt-125m")
    assert cfg.model == "facebook/opt-125m"
    assert cfg.max_model_len == 4096
    assert cfg.dtype == "bfloat16"


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
