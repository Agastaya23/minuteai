"""ROUGE-1/2/L and (optionally) BERTScore against gold abstractive summaries."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def rouge(prediction: str, reference: str) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeLsum"].fmeasure,
    }


def bertscore(predictions: list[str], references: list[str]) -> list[float] | None:
    """BERTScore F1 per pair, or None if bert-score isn't installed."""
    try:
        from bert_score import score
    except ImportError:
        log.info("bert-score not installed (uv sync --extra bertscore); skipping")
        return None
    _, _, f1 = score(predictions, references, lang="en", rescale_with_baseline=True,
                     verbose=False)
    return [float(x) for x in f1]


def evaluate(pairs: list[tuple[str, str]], with_bertscore: bool = False) -> dict[str, float]:
    """Mean scores over (prediction, reference) pairs; references joined with newlines."""
    if not pairs:
        return {}
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in pairs:
        for k, v in rouge(pred, ref).items():
            totals[k] += v
    out = {k: v / len(pairs) for k, v in totals.items()}
    if with_bertscore:
        bs = bertscore([p for p, _ in pairs], [r for _, r in pairs])
        if bs is not None:
            out["bertscore_f1"] = sum(bs) / len(bs)
    out["n"] = len(pairs)
    return out
