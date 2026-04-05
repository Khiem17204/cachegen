"""Utility to visualize benchmark results from benchmarks/results.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _pd():
    import pandas as pd

    return pd


def load_results(path: Path) -> "pd.DataFrame":
    pd = _pd()
    data = json.loads(path.read_text())
    df = pd.DataFrame(data["runs"])
    if df.empty:
        raise ValueError("results.json contains no runs")

    if "mode" not in df.columns:
        df["mode"] = "single"
    if "bandwidth_mbps" not in df.columns:
        df["bandwidth_mbps"] = 1000.0

    return df


def _plot_grouped_bar(
    df: "pd.DataFrame",
    value_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt = _plt()
    pivot = df.pivot_table(index="model", columns="mode", values=value_col, aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(9, 4.5))
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Mode", fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ttft(df: "pd.DataFrame", out_dir: Path) -> None:
    _plot_grouped_bar(
        df=df,
        value_col="ttft_seconds",
        ylabel="TTFT (s)",
        title="Average TTFT by Model/Mode",
        out_path=out_dir / "ttft_by_model_mode.png",
    )


def plot_throughput(df: "pd.DataFrame", out_dir: Path) -> None:
    _plot_grouped_bar(
        df=df,
        value_col="tokens_per_second",
        ylabel="Tokens / second",
        title="Average Throughput by Model/Mode",
        out_path=out_dir / "throughput_by_model_mode.png",
    )


def plot_compression_ratio(df: "pd.DataFrame", out_dir: Path) -> None:
    if "compression_ratio" not in df.columns:
        return
    _plot_grouped_bar(
        df=df,
        value_col="compression_ratio",
        ylabel="Compression Ratio (raw / compressed)",
        title="Average Compression Ratio by Model/Mode",
        out_path=out_dir / "compression_ratio_by_mode.png",
    )


def plot_network_time(df: "pd.DataFrame", out_dir: Path) -> None:
    plt = _plt()
    if "network_time" not in df.columns:
        return

    agg = (
        df.groupby(["bandwidth_mbps", "model", "mode"], as_index=False)["network_time"]
        .mean()
        .sort_values(["model", "mode", "bandwidth_mbps"])
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (model, mode), group in agg.groupby(["model", "mode"]):
        ax.plot(
            group["bandwidth_mbps"],
            group["network_time"],
            marker="o",
            label=f"{model} | {mode}",
        )

    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel("Network Time (s)")
    ax.set_title("Average Simulated Network Time by Bandwidth/Mode")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "network_time_by_mode_bandwidth.png")
    plt.close(fig)


def main() -> None:
    results_path = Path("benchmarks/results.json")
    if not results_path.exists():
        raise FileNotFoundError("Run run_benchmarks.py first to produce benchmarks/results.json")

    out_dir = results_path.parent
    df = load_results(results_path)

    plot_ttft(df, out_dir)
    plot_throughput(df, out_dir)
    plot_compression_ratio(df, out_dir)
    plot_network_time(df, out_dir)
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
