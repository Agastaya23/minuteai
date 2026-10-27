"""Commitment lifecycle classification: accuracy + confusion matrix vs manual labels.

Labels CSV (eval/data/commitment_labels.csv): commitment_id,gold_state
"""

from __future__ import annotations

import csv
from pathlib import Path

STATES = ["PROPOSED", "COMMITTED", "FULFILLED", "SLIPPED", "SILENTLY_DROPPED", "REVERSED"]


def load_labels(path: Path) -> dict[str, str]:
    with path.open() as f:
        return {r["commitment_id"]: r["gold_state"].strip().upper() for r in csv.DictReader(f)}


def evaluate(predicted: dict[str, str], gold: dict[str, str]) -> dict:
    ids = [i for i in gold if i in predicted]
    matrix = {g: {p: 0 for p in STATES} for g in STATES}
    for i in ids:
        matrix.setdefault(gold[i], {p: 0 for p in STATES})
        matrix[gold[i]][predicted[i]] = matrix[gold[i]].get(predicted[i], 0) + 1
    correct = sum(predicted[i] == gold[i] for i in ids)
    per_state = {}
    for s in STATES:
        tp = matrix.get(s, {}).get(s, 0)
        pred_s = sum(row.get(s, 0) for row in matrix.values())
        gold_s = sum(matrix.get(s, {}).values())
        p = tp / pred_s if pred_s else 0.0
        r = tp / gold_s if gold_s else 0.0
        per_state[s] = {"precision": p, "recall": r, "support": gold_s,
                        "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return {"n": len(ids), "accuracy": correct / len(ids) if ids else float("nan"),
            "confusion": matrix, "per_state": per_state,
            "unlabelled_predictions": len(set(predicted) - set(gold))}


def write_label_template(predicted: dict[str, str], descriptions: dict[str, str], path: Path
                         ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["commitment_id", "description", "predicted_state", "gold_state"])
        for cid, state in predicted.items():
            w.writerow([cid, descriptions.get(cid, ""), state, ""])
