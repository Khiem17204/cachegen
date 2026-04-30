import json
from pathlib import Path

import pytest

from visualize_colab_ttft_slowdown import (
    OUTPUT_FILENAME,
    load_summary,
    plot_slowdown_figure,
)


def _write_summary(path: Path, *, fp8: dict[tuple[int, float], float], cachegen: dict[tuple[int, float], float], note: str = "") -> None:
    rows = []
    for (prompt_length, bandwidth), value in sorted(fp8.items()):
        rows.append(
            {
                "mode": "quantized_fp8",
                "bandwidth_mbps": bandwidth,
                "prompt_length_tokens_requested": prompt_length,
                "median_ttft_seconds": value,
            }
        )
    for (prompt_length, bandwidth), value in sorted(cachegen.items()):
        rows.append(
            {
                "mode": "cachegen",
                "bandwidth_mbps": bandwidth,
                "prompt_length_tokens_requested": prompt_length,
                "median_ttft_seconds": value,
            }
        )

    payload = {
        "meta": {
            "gpu_name": "Test GPU",
            "model": "test/model",
            "note": note,
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload))


def test_load_summary_requires_rows(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"meta": {}}))

    with pytest.raises(ValueError, match="non-empty rows"):
        load_summary(path)


def test_plot_slowdown_figure_emits_png(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib", exc_type=ImportError)
    matplotlib.use("Agg")

    fp8 = {
        (2048, 100.0): 0.13,
        (2048, 3000.0): 0.13,
        (4096, 100.0): 0.27,
        (4096, 3000.0): 0.27,
    }
    naive = {
        (2048, 100.0): 10.0,
        (2048, 3000.0): 3.2,
        (4096, 100.0): 20.0,
        (4096, 3000.0): 6.4,
    }
    optimized = {
        (2048, 100.0): 0.20,
        (2048, 3000.0): 0.15,
        (4096, 100.0): 0.40,
        (4096, 3000.0): 0.31,
    }

    naive_path = tmp_path / "naive.json"
    optimized_path = tmp_path / "optimized.json"
    _write_summary(naive_path, fp8=fp8, cachegen=naive)
    _write_summary(optimized_path, fp8=fp8, cachegen=optimized, note="Illustrative example only")

    output_path = plot_slowdown_figure(
        load_summary(naive_path),
        load_summary(optimized_path),
        tmp_path,
    )

    assert output_path == tmp_path / OUTPUT_FILENAME
    assert output_path.exists()
