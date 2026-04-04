"""Stress test simulator for CacheGen pipeline.

Runs repeated encode/decode cycles across high seq lengths and varying bandwidths
to emulate pressure scenarios without launching a vLLM server.
"""
from __future__ import annotations

import random
import time
from typing import List

import torch

from vllm_integration import NetworkSimulator, PhysicalBlock, VllmCacheGenHook, MockVllmCacheEngine

BANDWIDTHS = [50, 100, 200]  # Mbps
SEQ_LENS = [1024, 2048, 4096]
NUM_RUNS = 10


def run_once(seq_len: int, bandwidth: int) -> float:
    block = torch.randn(1, 32, seq_len, 128, dtype=torch.float16)
    hook = VllmCacheGenHook(
        cache_engine=MockVllmCacheEngine(), network_simulator=NetworkSimulator(bandwidth)
    )
    start = time.perf_counter()
    hook.offload_and_insert([PhysicalBlock(block, block_id=0)])
    return time.perf_counter() - start


def main() -> None:
    latencies: List[float] = []
    for _ in range(NUM_RUNS):
        bw = random.choice(BANDWIDTHS)
        seq = random.choice(SEQ_LENS)
        lat = run_once(seq, bw)
        latencies.append(lat)
        print(f"run bw={bw}Mbps seq={seq} -> {lat:.3f}s")
    print(f"p50={torch.tensor(latencies).quantile(0.5):.3f}s p95={torch.tensor(latencies).quantile(0.95):.3f}s")


if __name__ == "__main__":
    main()
