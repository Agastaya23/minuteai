"""Action-item detection and owner-assignment metrics.

Predicted items are matched one-to-one to gold items by semantic similarity
(greedy on the similarity matrix). Owner assignment is reported as three
separate rates over matched items with a gold owner, because conflating them
hides the contribution:

  owner_accuracy        predicted owner == gold owner
  abstention_rate       predicted owner is None (system said "unassigned")
  wrong_assignment_rate predicted a *different* person than gold

plus `unowned_correct_abstention`: gold has no owner and the system abstained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minuteai.graph.similarity import Similarity
from minuteai.models.artifacts import ActionItem
from minuteai.search.embedder import Embedder, get_embedder
from minuteai.storage.schema import normalize_name

MATCH_THRESHOLD = 0.5


@dataclass
class Match:
    pred: int
    gold: int
    similarity: float


def match_items(pred: list[str], gold: list[str], embedder: Embedder | None = None,
                threshold: float = MATCH_THRESHOLD) -> list[Match]:
    if not pred or not gold:
        return []
    sim = Similarity(embedder or get_embedder())
    m = np.array([[sim.score(p, g) for g in gold] for p in pred])
    matches: list[Match] = []
    used_p, used_g = set(), set()
    for flat in np.argsort(-m, axis=None):
        i, j = divmod(int(flat), len(gold))
        if m[i, j] < threshold:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append(Match(i, j, float(m[i, j])))
    return matches


def _same_person(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    na, nb = normalize_name(a), normalize_name(b)
    return na == nb or na.split("_")[0] == nb.split("_")[0]


def evaluate(
    predicted: list[ActionItem],
    gold: list[str],
    gold_owners: dict[int, str | None] | None = None,
    embedder: Embedder | None = None,
) -> dict[str, float]:
    matches = match_items([p.description for p in predicted], gold, embedder)
    tp = len(matches)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {"n_pred": len(predicted), "n_gold": len(gold), "tp": tp,
           "precision": precision, "recall": recall, "f1": f1}
    if gold_owners:
        owned = [m for m in matches if gold_owners.get(m.gold)]
        unowned = [m for m in matches if m.gold in gold_owners and not gold_owners[m.gold]]
        correct = sum(_same_person(predicted[m.pred].owner, gold_owners[m.gold]) for m in owned)
        abstained = sum(predicted[m.pred].owner is None for m in owned)
        wrong = len(owned) - correct - abstained
        n = len(owned) or 1
        out.update({
            "owner_n": len(owned),
            "owner_accuracy": correct / n,
            "abstention_rate": abstained / n,
            "wrong_assignment_rate": wrong / n,
            "unowned_n": len(unowned),
            "unowned_correct_abstention": (
                sum(predicted[m.pred].owner is None for m in unowned) / len(unowned)
                if unowned else float("nan")),
        })
    return out
