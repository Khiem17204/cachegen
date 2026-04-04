"""Utility to visualize benchmark results.

Reads benchmarks/results.json produced by run_benchmarks.py and writes simple
PNG charts comparing baseline vs CacheGen across bandwidths and prompt sizes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path: Path):
    data = json.loads(path.read_text())
    return data, pd.DataFrame(data["runs"])


def plot_latency(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for (mode, model), group in df.groupby(["mode", "model"]):
        group = group.sort_values("bandwidth_mbps")
        ax.plot(group["bandwidth_mbps"], group["ttft"], marker="o", label=f"{mode}-{model}")
    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("TTFT vs Bandwidth")
    ax.legend()
    out = out_dir / "ttft_vs_bw.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_compression(df: pd.DataFrame, out_dir: Path) -> None:
    comp = df[df["mode"] == "cachegen"][
        ["prompt_name", "bandwidth_mbps", "compression_ratio", "model"]
    ]
    comp = comp.sort_values(["model", "prompt_name", "bandwidth_mbps"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for (model, prompt_name), group in comp.groupby(["model", "prompt_name"]):
        ax.plot(group["bandwidth_mbps"], group["compression_ratio"], marker="s", label=f"{model}-{prompt_name}")
    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel("Compression Ratio (raw / compressed)")
    ax.set_title("CacheGen Compression Ratio")
    ax.legend()
    out = out_dir / "compression_ratio.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_quality(data: dict, out_dir: Path) -> None:
    if "quality" not in data:
        return
    q = data["quality"]
    metrics = ["bleu_diff", "rouge1_diff", "ppl_diff"]
    vals = [q.get(m, 0.0) for m in metrics]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(metrics, vals)
    ax.set_title("Quality deltas (CacheGen - Baseline)")
    out = out_dir / "quality_deltas.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    results_path = Path("benchmarks/results.json")
    if not results_path.exists():
        raise FileNotFoundError("Run run_benchmarks.py first to produce benchmarks/results.json")

    out_dir = results_path.parent
    data, df = load_results(results_path)

    plot_latency(df, out_dir)
    plot_compression(df, out_dir)
    plot_quality(data, out_dir)
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
