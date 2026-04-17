# vLLM Integration (Shim)

## Objective

Demonstrate a shim around vLLM-like physical blocks without modifying upstream vLLM code. This module prepares per-block tensors for the CacheGen encoder/decoder, simulates network transfer, then reinserts decoded blocks into a mock cache engine.

## Components

| Class | Purpose |
| --- | --- |
| `PhysicalBlock` | Holder for a single vLLM physical block shaped `[num_layers, num_heads, block_size, head_dim]`. |
| `NetworkSimulator` | Applies a sleep-based delay using `seconds = bytes * 8 / (bandwidth_mbps * 1_000_000)`. |
| `MockVllmCacheEngine` | Minimal cache store mirroring `CacheEngine.insert/get` semantics for tests. |
| `VllmCacheGenHook` | Glue layer: duplicates each block across a synthetic K/V axis, runs CacheGen encoder/decoder, simulates transfer, and inserts decoded blocks back into the cache engine. |

## Usage Example

```python
import torch
from vllm_integration import (
    PhysicalBlock, NetworkSimulator, MockVllmCacheEngine, VllmCacheGenHook,
)

blocks = [PhysicalBlock(torch.randn(1, 4, 8, 8, dtype=torch.float16), block_id=0)]
network = NetworkSimulator(bandwidth_mbps=100)
cache = MockVllmCacheEngine()
shim = VllmCacheGenHook(cache_engine=cache, network_simulator=network)

stats = shim.offload_and_insert(blocks)
print(stats)
restored = cache.get(0)
```

## Swapping Standard Cache with CacheGen (Simulation)

This package is a simulation shim, not a real vLLM cache-engine integration. `VllmCacheGenHook` works by duplicating a 4-D physical block into a synthetic `[L, 2, H, S, D]` tensor so it can reuse the current CacheGen pipeline, then dropping that synthetic axis after decode.

1. Replace your normal `CacheEngine.insert` call in a test or prototype with `VllmCacheGenHook.offload_and_insert(blocks)`. The hook handles encode → simulated transfer → decode per block.
2. Use `NetworkSimulator` to set the target link bandwidth; stats returned include `compressed_bytes`, `network_time`, and `decode_time` for accounting.
3. Treat the result as a demonstration of block-by-block wrapping behavior, not as proof that the repository has direct integration with vLLM internals, paged-attention kernels, or production cache management.

## Source of Truth

If this README and runtime behavior diverge, treat `vllm_integration/hook.py` and `tests/test_vllm_integration.py` as authoritative.

## Testing

```
pytest tests/test_vllm_integration.py -q
```
