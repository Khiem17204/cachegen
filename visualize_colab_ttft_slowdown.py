"""Render a merged slowdown-ratio figure from two Colab TTFT summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NAIVE_INPUT_FILENAME = "colab_ttft_nonoptimized.json"
OPTIMIZED_INPUT_FILENAME = "colab_ttft_optimzed.json"
OUTPUT_FILENAME = "colab_ttft_slowdown_merged.png"


def _plt():
    import matplotlib.pyplot as plt

    return plt


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "rows" not in data or not data["rows"]:
        raise ValueError(f"summary JSON must contain non-empty rows: {path}")
    return data


def _ratio_rows(summary: dict[str, Any]) -> dict[tuple[int, float], float]:
    rows = summary["rows"]
    baseline = {
        (int(row["prompt_length_tokens_requested"]), float(row["bandwidth_mbps"])): float(
            row["median_ttft_seconds"]
        )
        for row in rows
        if row["mode"] == "quantized_fp8"
    }
    cachegen = {
        (int(row["prompt_length_tokens_requested"]), float(row["bandwidth_mbps"])): float(
            row["median_ttft_seconds"]
        )
        for row in rows
        if row["mode"] == "cachegen"
    }

    keys = sorted(set(baseline).intersection(cachegen))
    if not keys:
        raise ValueError("no matching quantized_fp8/cachegen rows found")

    return {key: cachegen[key] / baseline[key] for key in keys}


def plot_slowdown_figure(
    naive_summary: dict[str, Any],
    optimized_summary: dict[str, Any],
    out_dir: Path,
) -> Path:
    plt = _plt()
    from matplotlib.ticker import FuncFormatter, LogLocator

    naive_ratios = _ratio_rows(naive_summary)
    optimized_ratios = _ratio_rows(optimized_summary)
    prompt_lengths = sorted({prompt for prompt, _ in naive_ratios})
    bandwidths = sorted({bandwidth for _, bandwidth in naive_ratios})

    fig, axes = plt.subplots(1, len(prompt_lengths), figsize=(15.5, 5.4), sharey=True)
    if len(prompt_lengths) == 1:
        axes = [axes]

    series_specs = [
        ("naive decode", naive_ratios, "#d94f4f", "s"),
        ("optimized decode", optimized_ratios, "#386fa4", "o"),
    ]

    def format_ratio(value: float, _: float) -> str:
        if value < 1:
            return ""
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return f"{rounded}x"
        return f"{value:g}x"

    for ax, prompt_length in zip(axes, prompt_lengths, strict=True):
        for label, ratios, color, marker in series_specs:
            y_values = [ratios[(prompt_length, bandwidth)] for bandwidth in bandwidths]
            ax.plot(
                bandwidths,
                y_values,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.6,
                markersize=6.5,
            )

        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.4, alpha=0.9)
        ax.set_title(f"{prompt_length:,} prompt tokens", fontsize=18, pad=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(bandwidths, [str(int(value)) for value in bandwidths])
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
        ax.yaxis.set_major_formatter(FuncFormatter(format_ratio))
        ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.24, linewidth=1.0)
        ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.12, linewidth=0.9)
        ax.set_xlabel("Bandwidth (Mbps)", fontsize=15, labelpad=8)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_facecolor("#fbfbfb")

    axes[0].text(
        bandwidths[0],
        1.08,
        "Parity",
        color="#555555",
        fontsize=11,
        ha="left",
        va="bottom",
    )

    fig.supylabel("Slowdown Ratio (CacheGen TTFT / FP8 TTFT)", fontsize=17, x=0.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        fontsize=17,
    )

    meta = naive_summary.get("meta", {})
    model = str(meta.get("model", "unknown model"))
    gpu_name = str(meta.get("gpu_name", "unknown GPU"))
    fig.suptitle(
        f"Colab TTFT Slowdown Comparison on {gpu_name}: {model}\n"
        "Naive decode is much slower; optimized decode moves closer to parity but remains above 1.0",
        fontsize=18,
        fontweight="bold",
        y=1.06,
    )

    optimized_note = str(optimized_summary.get("meta", {}).get("note", "")).strip()
    if optimized_note:
        fig.text(
            0.5,
            0.935,
            f"Optimized series note: {optimized_note}",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#8b1e1e",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#fff3cd", "edgecolor": "#e0a800"},
        )

    fig.tight_layout(rect=(0.03, 0.02, 1, 0.86))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / OUTPUT_FILENAME
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(
    naive_path: Path | None = None,
    optimized_path: Path | None = None,
) -> None:
    naive_path = Path("benchmarks") / NAIVE_INPUT_FILENAME if naive_path is None else naive_path
    optimized_path = (
        Path("benchmarks") / OPTIMIZED_INPUT_FILENAME if optimized_path is None else optimized_path
    )
    if not naive_path.exists():
        raise FileNotFoundError(f"Naive summary file not found: {naive_path}")
    if not optimized_path.exists():
        raise FileNotFoundError(f"Optimized summary file not found: {optimized_path}")

    naive_summary = load_summary(naive_path)
    optimized_summary = load_summary(optimized_path)
    output_path = plot_slowdown_figure(naive_summary, optimized_summary, naive_path.parent)
    print(f"Wrote plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a merged TTFT slowdown figure")
    parser.add_argument("naive_path", nargs="?", type=Path)
    parser.add_argument("optimized_path", nargs="?", type=Path)
    args = parser.parse_args()
    main(args.naive_path, args.optimized_path)
