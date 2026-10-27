"""Retrieval metrics: hit@k and MRR against gold relevant turn spans (QMSum)."""

from __future__ import annotations


def is_relevant(start_utt: int, end_utt: int, spans: list[tuple[int, int]]) -> bool:
    return any(start_utt <= b and end_utt >= a for a, b in spans)


def evaluate(results: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]],
             ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    """results: per query, (retrieved [(start_utt, end_utt)] in rank order, gold spans)."""
    if not results:
        return {}
    hits = {k: 0 for k in ks}
    rr = 0.0
    for retrieved, gold in results:
        rank = next((i for i, (a, b) in enumerate(retrieved) if is_relevant(a, b, gold)), None)
        if rank is not None:
            rr += 1 / (rank + 1)
            for k in ks:
                hits[k] += rank < k
    n = len(results)
    return {"n": n, "mrr": rr / n, **{f"hit@{k}": hits[k] / n for k in ks}}
