import time

import pytest
import torch

from decoder.decoder import CacheGenDecoder
from encoder.encoder import CacheGenEncoder
from vllm_integration import (
    MockVllmCacheEngine,
    NetworkSimulator,
    PhysicalBlock,
    VllmCacheGenHook,
)


@pytest.mark.slow(reason="Simulates end-to-end block transfer timing")
def test_transfer_time_matches_bandwidth_and_decode_cost():
    torch.manual_seed(0)

    num_blocks = 1000
    block_shape = (1, 4, 8, 8)  # [num_layers, num_heads, block_size, head_dim]
    blocks = [
        PhysicalBlock(torch.randn(*block_shape, dtype=torch.float16), block_id=i)
        for i in range(num_blocks)
    ]

    bandwidth_mbps = 100
    encoder = CacheGenEncoder(chunk_size=block_shape[2])
    decoder = CacheGenDecoder()
    cache_engine = MockVllmCacheEngine()
    simulator = NetworkSimulator(bandwidth_mbps)

    hook = VllmCacheGenHook(
        cache_engine=cache_engine,
        network_simulator=simulator,
        encoder=encoder,
        decoder=decoder,
    )

    wall_start = time.perf_counter()
    stats = hook.offload_and_insert(blocks)
    wall_elapsed = time.perf_counter() - wall_start

    expected_transfer = stats["compressed_bytes"] * 8 / (bandwidth_mbps * 1_000_000)
    expected_total = expected_transfer + stats["decode_time"]

    # Allow modest overhead for Python scheduling and encode cost.
    assert wall_elapsed == pytest.approx(expected_total, rel=0.25, abs=0.1)

    # Data path sanity: ensure blocks landed in cache and survive round trip.
    assert len(cache_engine) == num_blocks
    assert torch.allclose(cache_engine.get(0), blocks[0].tensor, atol=5e-2)
