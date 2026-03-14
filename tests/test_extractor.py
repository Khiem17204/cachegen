"""
Test suite for KVCacheExtractor.

Uses GPT-2 (124M params) as the test model — small, fast to download,
and representative of the HuggingFace causal-LM interface.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_extraction_hf import KVCacheExtractor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_NAME = "gpt2"

# GPT-2 architecture constants
GPT2_NUM_LAYERS = 12
GPT2_NUM_HEADS = 12
GPT2_HEAD_DIM = 64


@pytest.fixture(scope="module")
def extractor():
    """Load GPT-2 once and share across all tests in this module."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return KVCacheExtractor(model, tokenizer, device="cpu")


@pytest.fixture(scope="module")
def tokenizer():
    """Standalone tokenizer for computing expected seq_len."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKVCacheExtractor:
    """Core test suite for the KVCacheExtractor."""

    PROMPT = "The quick brown fox jumps over the lazy dog."

    def test_output_shape(self, extractor, tokenizer):
        """Output tensor must be 5-D with correct layer/head/dim counts."""
        tensor = extractor.extract(self.PROMPT)
        seq_len = len(tokenizer.encode(self.PROMPT))

        assert tensor.ndim == 5, f"Expected 5-D tensor, got {tensor.ndim}-D"
        assert tensor.shape == (
            GPT2_NUM_LAYERS,
            2,
            GPT2_NUM_HEADS,
            seq_len,
            GPT2_HEAD_DIM,
        ), f"Unexpected shape: {tensor.shape}"

    def test_output_dtype(self, extractor):
        """Output must be FP16."""
        tensor = extractor.extract(self.PROMPT)
        assert tensor.dtype == torch.float16, f"Expected float16, got {tensor.dtype}"

    def test_nonempty_values(self, extractor):
        """Extraction must produce non-zero values (i.e., it actually ran)."""
        tensor = extractor.extract(self.PROMPT)
        assert tensor.abs().sum().item() > 0, "Tensor is all zeros"

    def test_contiguous(self, extractor):
        """Output tensor must be contiguous in memory."""
        tensor = extractor.extract(self.PROMPT)
        assert tensor.is_contiguous(), "Tensor is not contiguous"

    @pytest.mark.parametrize(
        "prompt",
        [
            "Hello",
            "The quick brown fox jumps.",
            (
                "In a distant galaxy far far away, there lived a civilization "
                "that had mastered the art of compressing knowledge into tiny "
                "crystalline structures no larger than a grain of sand."
            ),
        ],
        ids=["short", "medium", "long"],
    )
    def test_varying_prompt_lengths(self, extractor, tokenizer, prompt):
        """seq_len dimension must match the tokenized prompt length."""
        tensor = extractor.extract(prompt)
        expected_seq_len = len(tokenizer.encode(prompt))

        assert tensor.shape[3] == expected_seq_len, (
            f"seq_len mismatch: tensor has {tensor.shape[3]}, "
            f"expected {expected_seq_len}"
        )
        # Also verify the other dimensions are still correct
        assert tensor.shape[:3] == (GPT2_NUM_LAYERS, 2, GPT2_NUM_HEADS)
        assert tensor.shape[4] == GPT2_HEAD_DIM
