# KV Cache Extraction

## Objective

Extract raw FP16 KV cache tensors from any HuggingFace causal language model in a single prefill pass, producing a standardised 5-D tensor ready for downstream CacheGen compression.

## API Reference

### `KVCacheExtractor`

```python
class KVCacheExtractor:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None: ...

    def extract(self, prompt: str) -> torch.Tensor: ...
```

| Method | Returns |
|---|---|
| `extract(prompt)` | `torch.Tensor` of shape `[num_layers, 2, num_kv_heads, seq_len, head_dim]`, dtype `float16` |

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_extraction_hf import KVCacheExtractor

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

extractor = KVCacheExtractor(model, tokenizer, device="cpu")
kv_cache = extractor.extract("The quick brown fox")

print(kv_cache.shape)   # [12, 2, 12, 5, 64]
print(kv_cache.dtype)   # torch.float16
```

## Testing

```bash
# Run the extractor test suite (downloads GPT-2 ~500 MB on first run)
pytest tests/test_extractor.py -v
```
