"""Quality evaluation helpers for CacheGen vs baseline outputs.

If sacrebleu/rouge_score are available, compute standard metrics; otherwise
fall back to simple token-overlap approximations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    import sacrebleu  # type: ignore
except Exception:  # pragma: no cover
    sacrebleu = None

try:
    from rouge_score import rouge_scorer  # type: ignore
except Exception:  # pragma: no cover
    rouge_scorer = None


def _simple_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    overlap = sum(1 for t in hyp_tokens if t in ref_tokens)
    return overlap / max(len(hyp_tokens), 1)


def _simple_rouge1(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in hyp_tokens:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1
    precision = overlap / max(len(hyp_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _simple_perplexity(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    mismatch = abs(len(ref_tokens) - len(hyp_tokens)) + sum(
        1 for r, h in zip(ref_tokens, hyp_tokens) if r != h
    )
    return 1.0 + mismatch / max(len(ref_tokens), 1)


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    if sacrebleu:
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score / 100.0
    else:
        bleu = _simple_bleu(reference, hypothesis)

    if rouge_scorer:
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        rouge1 = scorer.score(reference, hypothesis)["rouge1"].fmeasure
    else:
        rouge1 = _simple_rouge1(reference, hypothesis)

    ppl = _simple_perplexity(reference, hypothesis)
    return {"bleu": bleu, "rouge1": rouge1, "perplexity": ppl}


def aggregate_diffs(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"bleu_diff": 0.0, "rouge1_diff": 0.0, "ppl_diff": 0.0}
    bleu_diff = sum(r["bleu_diff"] for r in rows) / len(rows)
    rouge_diff = sum(r["rouge1_diff"] for r in rows) / len(rows)
    ppl_diff = sum(r["ppl_diff"] for r in rows) / len(rows)
    return {"bleu_diff": bleu_diff, "rouge1_diff": rouge_diff, "ppl_diff": ppl_diff}
