import json
from pathlib import Path

import pytest

from visualize_results import (
    load_results,
    plot_compression_ratio,
    plot_network_time,
    plot_throughput,
    plot_ttft,
)


def _write_results(path: Path) -> None:
    payload = {
        "runs": [
            {
                "model": "facebook/opt-125m",
                "mode": "baseline",
                "prompt_name": "medium",
                "bandwidth_mbps": 100.0,
                "ttft_seconds": 0.20,
                "tokens_per_second": 120.0,
                "compression_ratio": 1.0,
                "network_time": 0.015,
            },
            {
                "model": "facebook/opt-125m",
                "mode": "cachegen",
                "prompt_name": "medium",
                "bandwidth_mbps": 100.0,
                "ttft_seconds": 0.16,
                "tokens_per_second": 130.0,
                "compression_ratio": 1.8,
                "network_time": 0.009,
            },
            {
                "model": "facebook/opt-125m",
                "mode": "baseline",
                "prompt_name": "long",
                "bandwidth_mbps": 1000.0,
                "ttft_seconds": 0.08,
                "tokens_per_second": 150.0,
                "compression_ratio": 1.0,
                "network_time": 0.003,
            },
            {
                "model": "facebook/opt-125m",
                "mode": "cachegen",
                "prompt_name": "long",
                "bandwidth_mbps": 1000.0,
                "ttft_seconds": 0.07,
                "tokens_per_second": 160.0,
                "compression_ratio": 1.7,
                "network_time": 0.002,
            },
        ]
    }
    path.write_text(json.dumps(payload))


def test_load_results_adds_defaults(tmp_path: Path) -> None:
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
    plot_throughput(df, tmp_path)
    plot_compression_ratio(df, tmp_path)
    plot_network_time(df, tmp_path)

    assert (tmp_path / "ttft_by_model_mode.png").exists()
    assert (tmp_path / "throughput_by_model_mode.png").exists()
    assert (tmp_path / "compression_ratio_by_mode.png").exists()
    assert (tmp_path / "network_time_by_mode_bandwidth.png").exists()
