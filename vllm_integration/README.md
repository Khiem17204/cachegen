# vLLM Integration (Shim)

## Objective

Demonstrate how CacheGen's Phase A compression pipeline can wrap vLLM's paged-attention blocks without modifying upstream vLLM code. The shim encodes per-physical-block tensors, simulates network transfer, then decodes and reinserts them into a cache engine.

## Components

| Class | Purpose |
| --- | --- |
| `PhysicalBlock` | Holder for a single vLLM physical block shaped `[num_layers, num_heads, block_size, head_dim]`. |
| `NetworkSimulator` | Applies a sleep-based delay using `seconds = bytes * 8 / (bandwidth_mbps * 1_000_000)`. |
| `MockVllmCacheEngine` | Minimal cache store mirroring `CacheEngine.insert/get` semantics for tests. |
| `VllmCacheGenHook` | Glue layer: prepares blocks with a synthetic K/V axis, runs CacheGen encoder/decoder, simulates transfer, and inserts decoded blocks back into the cache engine. |

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

1. Replace your normal `CacheEngine.insert` call with `VllmCacheGenHook.offload_and_insert(blocks)`. The hook handles encode → transfer → decode per block.
2. Use `NetworkSimulator` to set the target link bandwidth; stats returned include `compressed_bytes`, `network_time`, and `decode_time` for accounting.
3. If integrating into real vLLM, keep BlockManager intact: adapt your worker to supply `PhysicalBlock` tensors and delegate to the hook before insertion.

## Testing

```
pytest tests/test_vllm_integration.py -q
```
