import json
from pathlib import Path

import pytest

from visualize_results import (
    load_results,
    plot_ttft,
    plot_transport_breakdown,
)


def _write_results(path: Path) -> None:
    payload = {
        "meta": {
            "generated_at_iso": "2026-04-22T13:00:00-04:00",
        },
        "config": {
            "bandwidth_mbps": [3000.0],
        },
        "runs": [
            {
                "model": "mistralai/Mistral-7B-Instruct-v0.3",
                "mode": "quantized_fp8",
                "prompt_id": "len2048_sample1",
                "prompt_length_tokens_requested": 2048,
                "bandwidth_mbps": 3000.0,
                "ttft_seconds": 0.24,
                "tokens_per_second": 120.0,
                "transmitted_bytes": 240000.0,
                "network_time": 0.015,
                "decode_time": 0.006,
            },
            {
                "model": "mistralai/Mistral-7B-Instruct-v0.3",
                "mode": "cachegen",
                "prompt_id": "len2048_sample1",
                "prompt_length_tokens_requested": 2048,
                "bandwidth_mbps": 3000.0,
                "ttft_seconds": 0.09,
                "tokens_per_second": 130.0,
                "transmitted_bytes": 90000.0,
                "network_time": 0.009,
                "decode_time": 0.008,
            },
            {
                "model": "mistralai/Mistral-7B-Instruct-v0.3",
                "mode": "quantized_fp8",
                "prompt_id": "len4096_sample1",
                "prompt_length_tokens_requested": 4096,
                "bandwidth_mbps": 3000.0,
                "ttft_seconds": 0.42,
                "tokens_per_second": 150.0,
                "transmitted_bytes": 480000.0,
                "network_time": 0.031,
                "decode_time": 0.010,
            },
            {
                "model": "mistralai/Mistral-7B-Instruct-v0.3",
                "mode": "cachegen",
                "prompt_id": "len4096_sample1",
                "prompt_length_tokens_requested": 4096,
                "bandwidth_mbps": 3000.0,
                "ttft_seconds": 0.14,
                "tokens_per_second": 160.0,
                "transmitted_bytes": 150000.0,
                "network_time": 0.014,
                "decode_time": 0.012,
            },
        ],
        "summaries": {},
        "claim_check": {
            "ttft_speedup_3gbps": 2.8,
            "range_status": "below_paper_range",
        },
    }
    path.write_text(json.dumps(payload))


def test_load_results_reads_new_schema(tmp_path: Path) -> None:
    pytest.importorskip("pandas", exc_type=(ImportError, ValueError))

    path = tmp_path / "results.json"
    _write_results(path)

    df = load_results(path)
    assert "mode" in df.columns
    assert "bandwidth_mbps" in df.columns
    assert len(df) == 4


def test_plot_functions_emit_files(tmp_path: Path) -> None:
    pytest.importorskip("pandas", exc_type=(ImportError, ValueError))
    matplotlib = pytest.importorskip("matplotlib", exc_type=ImportError)
    matplotlib.use("Agg")

    path = tmp_path / "results.json"
    _write_results(path)

    df = load_results(path)

    plot_ttft(df, tmp_path)
    plot_transport_breakdown(df, tmp_path)

    assert (tmp_path / "ttft_comparison_3gbps.png").exists()
    assert (tmp_path / "transport_breakdown_3gbps.png").exists()
