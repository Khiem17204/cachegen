"""Stage 4: vLLM integration shims and simulation utilities.

This module avoids touching vLLM internals while showing how CacheGen's
Phase A components can be applied block-by-block inside a paged-attention
style cache manager.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch

from decoder.decoder import CacheGenDecoder
from encoder.encoder import CacheGenEncoder, EncodedChunk


@dataclass
class PhysicalBlock:
    """Represents a vLLM physical block.

    The shape matches paged attention allocations:
        [num_layers, num_heads, block_size, head_dim].
    """

    tensor: torch.Tensor
    block_id: Optional[int] = None


class NetworkSimulator:
    """Simulates network transfer time given a bandwidth budget.

    Bandwidth is interpreted as *megabits per second* (Mbps). The simulator
    simply sleeps for the computed duration to make wall-clock time match the
    theoretical transfer cost.
    """

    def __init__(self, bandwidth_mbps: float) -> None:
        if bandwidth_mbps <= 0:
            raise ValueError("bandwidth_mbps must be positive")
        self.bandwidth_mbps = float(bandwidth_mbps)

    def compute_delay(self, num_bytes: int) -> float:
        """Return seconds of transfer time for ``num_bytes`` at ``bandwidth_mbps``."""

        return (num_bytes * 8) / (self.bandwidth_mbps * 1_000_000)

    def transfer(self, payload: bytes) -> tuple[bytes, float]:
        """Simulate sending ``payload`` over the network.

        Returns the payload unchanged plus the delay applied.
        """

        delay = self.compute_delay(len(payload))
        time.sleep(delay)
        return payload, delay


class MockVllmCacheEngine:
    """Minimal stand-in for vLLM's CacheEngine.

    It stores decoded blocks by ``block_id`` and mimics insertion semantics
    without needing to import real vLLM code.
    """

    def __init__(self) -> None:
        self._store: dict[int, torch.Tensor] = {}

    def insert(self, block_id: int, tensor: torch.Tensor) -> None:
        self._store[block_id] = tensor

    def get(self, block_id: int) -> torch.Tensor:
        return self._store[block_id]

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self._store)


class VllmCacheGenHook:
    """Glue layer that maps vLLM physical blocks onto CacheGen's pipeline.

    This shim keeps vLLM unchanged: it prepares per-block tensors so they
    look like CacheGen's `[L, 2, H, S, D]` layout, runs the encoder/decoder,
    simulates network transfer, and reinserts the decoded block into a cache
    engine compatible with vLLM's expectations.
    """

    def __init__(
        self,
        cache_engine: MockVllmCacheEngine,
        network_simulator: NetworkSimulator,
        encoder: Optional[CacheGenEncoder] = None,
        decoder: Optional[CacheGenDecoder] = None,
    ) -> None:
        self.cache_engine = cache_engine
        self.network_simulator = network_simulator
        self.encoder = encoder or CacheGenEncoder()
        self.decoder = decoder or CacheGenDecoder()

    @staticmethod
    def _prepare_block(block: torch.Tensor) -> torch.Tensor:
        """Insert a fake K/V axis so CacheGen can operate per block."""

        if block.ndim != 4:
            raise ValueError(
                "Expected physical block of shape [L, H, S, D]; got " f"{block.shape}"
            )

        # Expand along the K/V axis (dim=1 in CacheGen layout) so encoder sees
        # [num_layers, 2, num_heads, seq_len, head_dim]. We duplicate data for
        # K and V channels to keep the mapping lossless for this simulation.
        return block.unsqueeze(1).expand(-1, 2, -1, -1, -1).contiguous()

    @staticmethod
    def _restore_block(decoded: torch.Tensor) -> torch.Tensor:
        """Drop the synthetic K/V axis after decoding."""

        if decoded.ndim != 5:
            raise ValueError(
                "Expected decoded tensor of shape [L, 2, H, S, D]; got "
                f"{decoded.shape}"
            )
        return decoded[:, 0].contiguous()

    def _encode_physical_block(self, block: torch.Tensor) -> List[EncodedChunk]:
        prepared = self._prepare_block(block)
        return self.encoder.encode(prepared)

    def _decode_physical_block(self, encoded_chunks: List[EncodedChunk]) -> torch.Tensor:
        decoded_full = self.decoder.decode(encoded_chunks)
        return self._restore_block(decoded_full)

    def offload_and_insert(self, blocks: Iterable[PhysicalBlock]) -> dict[str, float]:
        """Encode → simulate transfer → decode → insert for each block.

        Returns stats useful for timing assertions:
            compressed_bytes: total bytes sent over the simulated link.
            network_time: sum of per-chunk simulated delays.
            decode_time: cumulative decode wall-clock time.
        """

        stats = {"compressed_bytes": 0, "network_time": 0.0, "decode_time": 0.0}

        for idx, block in enumerate(blocks):
            block_id = block.block_id if block.block_id is not None else idx

            encoded_chunks = self._encode_physical_block(block.tensor)
            transmitted_chunks: list[EncodedChunk] = []

            for chunk in encoded_chunks:
                payload, delay = self.network_simulator.transfer(chunk.data)
                stats["network_time"] += delay
                stats["compressed_bytes"] += len(payload)

                transmitted_chunks.append(
                    EncodedChunk(
                        data=payload,
                        scales=chunk.scales,
                        original_dtype=chunk.original_dtype,
                        original_shape=chunk.original_shape,
                    )
                )

            decode_start = time.perf_counter()
            decoded_block = self._decode_physical_block(transmitted_chunks)
            stats["decode_time"] += time.perf_counter() - decode_start

            self.cache_engine.insert(block_id, decoded_block)

        return stats
