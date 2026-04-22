"""Utilities to visualize TTFT benchmark results from benchmarks/results.json."""

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

    return df


def _mode_order(df: "pd.DataFrame") -> list[str]:
    return list(dict.fromkeys(df["mode"].tolist()))


def _label_tuples(df: "pd.DataFrame") -> list[tuple[str, int]]:
    rows = (
        df[["model", "prompt_length_tokens_requested"]]
        .drop_duplicates()
        .sort_values(["model", "prompt_length_tokens_requested"])
    )
    return [
        (str(row["model"]), int(row["prompt_length_tokens_requested"]))
        for _, row in rows.iterrows()
    ]


def _label_strings(labels: list[tuple[str, int]]) -> list[str]:
    return [f"{model}\n{prompt_length} tokens" for model, prompt_length in labels]


def _median_rows(
    df: "pd.DataFrame",
    *,
    bandwidth_mbps: float,
    columns: list[str],
) -> "pd.DataFrame":
    filtered = df[df["bandwidth_mbps"] == bandwidth_mbps]
    if filtered.empty:
        raise ValueError(f"No runs found for bandwidth {bandwidth_mbps:g} Mbps")

    return (
        filtered.groupby(
            ["model", "prompt_length_tokens_requested", "mode"],
            as_index=False,
        )[columns]
        .median()
        .sort_values(["model", "prompt_length_tokens_requested", "mode"])
    )


def plot_ttft(
    df: "pd.DataFrame",
    out_dir: Path,
    *,
    bandwidth_mbps: float = 3000.0,
) -> None:
    plt = _plt()
    agg = _median_rows(df, bandwidth_mbps=bandwidth_mbps, columns=["ttft_seconds"])
    labels = _label_tuples(agg)
    label_strings = _label_strings(labels)
    modes = _mode_order(agg)
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    baseline_positions = list(range(len(labels)))
    for mode_index, mode in enumerate(modes):
        rows = agg[agg["mode"] == mode]
        offset = (mode_index - (len(modes) - 1) / 2.0) * width
        x_positions = [position + offset for position in baseline_positions]
        values: list[float] = []
        for label in labels:
            match = rows[
                (rows["model"] == label[0])
                & (rows["prompt_length_tokens_requested"] == label[1])
            ]
            values.append(float(match["ttft_seconds"].iloc[0]) if not match.empty else 0.0)
        ax.bar(x_positions, values, width=width, label=mode)

    ax.set_xticks(baseline_positions, label_strings)
    ax.set_xlabel("Model / Prompt Length")
    ax.set_ylabel("Median TTFT (s)")
    ax.set_title(f"TTFT Comparison at {bandwidth_mbps:g} Mbps")
    ax.legend(title="Mode", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ttft_comparison_3gbps.png")
    plt.close(fig)


def plot_transport_breakdown(
    df: "pd.DataFrame",
    out_dir: Path,
    *,
    bandwidth_mbps: float = 3000.0,
) -> None:
    plt = _plt()
    agg = _median_rows(
        df,
        bandwidth_mbps=bandwidth_mbps,
        columns=["transmitted_bytes", "network_time", "decode_time"],
    )
    labels = _label_tuples(agg)
    label_strings = _label_strings(labels)
    modes = _mode_order(agg)
    baseline_positions = list(range(len(labels)))
    width = 0.8 / max(len(modes), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_bytes, ax_time = axes

    for mode_index, mode in enumerate(modes):
        rows = agg[agg["mode"] == mode]
        offset = (mode_index - (len(modes) - 1) / 2.0) * width
        x_positions = [position + offset for position in baseline_positions]
        transmitted_values: list[float] = []
        network_values: list[float] = []
        decode_values: list[float] = []

        for label in labels:
            match = rows[
                (rows["model"] == label[0])
                & (rows["prompt_length_tokens_requested"] == label[1])
            ]
            if match.empty:
                transmitted_values.append(0.0)
                network_values.append(0.0)
                decode_values.append(0.0)
                continue
            transmitted_values.append(float(match["transmitted_bytes"].iloc[0]))
            network_values.append(float(match["network_time"].iloc[0]))
            decode_values.append(float(match["decode_time"].iloc[0]))

        ax_bytes.bar(x_positions, transmitted_values, width=width, label=mode)
        ax_time.bar(x_positions, network_values, width=width, label=f"{mode} network")
        ax_time.bar(
            x_positions,
            decode_values,
            width=width,
            bottom=network_values,
            label=f"{mode} decode",
        )

    ax_bytes.set_xticks(baseline_positions, label_strings)
    ax_bytes.set_xlabel("Model / Prompt Length")
    ax_bytes.set_ylabel("Median Transmitted Bytes")
    ax_bytes.set_title(f"Transfer Byte Breakdown at {bandwidth_mbps:g} Mbps")
    ax_bytes.legend(title="Mode", fontsize=8)

    ax_time.set_xticks(baseline_positions, label_strings)
    ax_time.set_xlabel("Model / Prompt Length")
    ax_time.set_ylabel("Median Time (s)")
    ax_time.set_title(f"Network vs Decode Time at {bandwidth_mbps:g} Mbps")
    ax_time.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "transport_breakdown_3gbps.png")
    plt.close(fig)


def main(results_path: Path | None = None) -> None:
    results_path = Path("benchmarks/results.json") if results_path is None else results_path
    if not results_path.exists():
        raise FileNotFoundError("Run run_benchmarks.py first to produce benchmarks/results.json")

    out_dir = results_path.parent
    df = load_results(results_path)

    plot_ttft(df, out_dir)
    plot_transport_breakdown(df, out_dir)
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
