"""vLLM CacheEngine wrapper that routes block insertions through CacheGen.

This keeps vLLM core untouched: wrap an existing CacheEngine and intercept
block insertions, compressing with CacheGen and simulating network transfer
before reinserting the decoded block back into the underlying engine.
"""
from __future__ import annotations

from typing import Any, Optional

try:  # vLLM is optional for this repo; wrapper is no-op without it.
    from vllm.core.block_manager import BlockSpaceManager  # type: ignore
    from vllm.core.kv_cache import CacheEngine  # type: ignore
except Exception:  # pragma: no cover - optional dep guard
    CacheEngine = object  # type: ignore
    BlockSpaceManager = object  # type: ignore

import torch

from .hook import NetworkSimulator, PhysicalBlock, VllmCacheGenHook


class CacheGenCacheEngine(CacheEngine):
    """Adapter around a vLLM CacheEngine that injects CacheGen encode/decode.

    Only the block insertion path is overridden; all other attributes/methods
    are forwarded to the wrapped ``inner`` engine.
    """

    def __init__(
        self,
        inner: CacheEngine,
        network_simulator: NetworkSimulator,
        encoder=None,
        decoder=None,
    ) -> None:
        self.inner = inner
        self.hook = VllmCacheGenHook(
            cache_engine=inner, network_simulator=network_simulator, encoder=encoder, decoder=decoder
        )

    # ---- core interception ----
    def insert_block(self, block_id: int, block: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Compress, simulate transfer, decode, then insert into the inner engine."""

        self.hook.offload_and_insert([PhysicalBlock(block, block_id=block_id)])

    # ---- attribute forwarding ----
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegation
        return getattr(self.inner, item)


def wrap_cache_engine(cache_engine: CacheEngine, bandwidth_mbps: float, encoder=None, decoder=None) -> CacheGenCacheEngine:
    """Helper to wrap an existing vLLM CacheEngine with CacheGen compression."""

    return CacheGenCacheEngine(cache_engine, NetworkSimulator(bandwidth_mbps), encoder=encoder, decoder=decoder)
