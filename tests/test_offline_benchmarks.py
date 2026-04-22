from types import SimpleNamespace

import torch

from run_offline_benchmarks import (
    OfflineBenchmarkConfig,
    WindowMetrics,
    build_report,
    build_token_windows,
    evaluate_windows,
)


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        use_cache: bool = False,
        past_key_values=None,
    ) -> SimpleNamespace:
        del use_cache
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        logits.scatter_(-1, (input_ids % self.vocab_size).unsqueeze(-1), 2.0)

        if past_key_values is None:
            past_key_values = tuple(
                (
                    torch.randn(1, 2, seq_len, 4, device=input_ids.device),
                    torch.randn(1, 2, seq_len, 4, device=input_ids.device),
                )
                for _ in range(2)
            )
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def test_build_token_windows_uses_deterministic_non_overlapping_offsets() -> None:
    windows = build_token_windows(list(range(12)), num_windows=3, window_length=4)

    assert windows == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]


def test_offline_quality_smoke_populates_finite_delta_fields() -> None:
    config = OfflineBenchmarkConfig(
        num_windows=1,
        window_length=8,
        cached_prefix_length=3,
        scored_continuation_length=4,
        chunk_size=2,
        compression_level=1,
    )

    report = evaluate_windows(TinyCausalLM(), [[1, 2, 3, 4, 5, 6, 7, 8]], config)
    aggregates = report["aggregates"]

    assert aggregates["windows"] == 1
    assert aggregates["total_scored_tokens"] == 4
    assert aggregates["compression_ratio"] > 0
    assert torch.isfinite(torch.tensor(aggregates["ppl_baseline"]))
    assert torch.isfinite(torch.tensor(aggregates["ppl_cachegen"]))
    assert "ppl_delta_abs" in aggregates
    assert "ppl_delta_pct" in aggregates
    assert report["windows"][0]["cachegen_payload_bytes"] > 0


def test_build_report_aggregates_compression_and_perplexity() -> None:
    config = OfflineBenchmarkConfig(num_windows=1)
    row = WindowMetrics(
        window_index=0,
        token_offset=0,
        cached_prefix_tokens=127,
        scored_tokens=2,
        raw_tensor_bytes=100,
        cachegen_payload_bytes=50,
        compression_ratio=2.0,
        baseline_nll=2.0,
        cachegen_nll=3.0,
        avg_nll_baseline=1.0,
        avg_nll_cachegen=1.5,
        ppl_baseline=2.718281828,
        ppl_cachegen=4.48168907,
        ppl_delta_abs=1.763407242,
        ppl_delta_pct=64.872,
    )

    report = build_report(config, [row])

    assert report["aggregates"]["compression_ratio"] == 2.0
    assert report["aggregates"]["avg_nll_baseline"] == 1.0
    assert report["aggregates"]["avg_nll_cachegen"] == 1.5
