"""Utility to visualize benchmark results.

Schemas:
- Live vLLM (run_benchmarks.py): columns include ``model``, ``mode`` (e.g., baseline/cachegen),
  ``ttft_seconds``, ``tokens_per_second``.
- Legacy simulated (removed artifacts): columns include ``mode`` and ``bandwidth_mbps``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    df = pd.DataFrame(data["runs"])
    if "mode" not in df.columns:
        df["mode"] = "single"
    return df


def plot_ttft_live(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot_table(index="prompt_name", columns=["model", "mode"], values="ttft_seconds")
    ax = pivot.plot(kind="bar", figsize=(8, 4))
    ax.set_xlabel("Prompt")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Time to First Token by Model/Mode")
    ax.legend(title="Model / Mode", fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    out = out_dir / "ttft_by_model.png"
    fig.savefig(out)
    plt.close(fig)


def plot_throughput_live(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot_table(index="prompt_name", columns=["model", "mode"], values="tokens_per_second")
    ax = pivot.plot(kind="bar", figsize=(8, 4))
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput by Model/Mode")
    ax.legend(title="Model / Mode", fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    out = out_dir / "throughput_by_model.png"
    fig.savefig(out)
    plt.close(fig)


def plot_ttft_sim(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, group in df.groupby("mode"):
        group = group.sort_values("bandwidth_mbps")
        ax.plot(group["bandwidth_mbps"], group["ttft"], marker="o", label=mode)
    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("TTFT vs Bandwidth (simulated)")
    ax.legend()
    out = out_dir / "ttft_vs_bw.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_compression_sim(df: pd.DataFrame, out_dir: Path) -> None:
    comp = df[df["mode"] == "cachegen"][["prompt_name", "bandwidth_mbps", "compression_ratio"]]
    comp = comp.sort_values(["prompt_name", "bandwidth_mbps"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for prompt_name, group in comp.groupby("prompt_name"):
        ax.plot(group["bandwidth_mbps"], group["compression_ratio"], marker="s", label=prompt_name)
    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel("Compression Ratio (raw / compressed)")
    ax.set_title("CacheGen Compression Ratio (simulated)")
    ax.legend()
    out = out_dir / "compression_ratio.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    results_path = Path("benchmarks/results.json")
    if not results_path.exists():
        raise FileNotFoundError("Run run_benchmarks.py first to produce benchmarks/results.json")

    out_dir = results_path.parent
    df = load_results(results_path)

    if "model" in df.columns:
        plot_ttft_live(df, out_dir)
        plot_throughput_live(df, out_dir)
        print(f"Wrote live plots to {out_dir}")
    elif "mode" in df.columns:
        plot_ttft_sim(df, out_dir)
        plot_compression_sim(df, out_dir)
        print(f"Wrote simulated plots to {out_dir}")
    else:
        raise ValueError("Unrecognized results schema; expected columns including 'model' or 'mode'")


if __name__ == "__main__":
    main()
