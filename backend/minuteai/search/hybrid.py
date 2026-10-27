"""Hybrid retrieval: BM25 over entry text + dense retrieval, fused with RRF."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from minuteai.models.search import SearchFilters
from minuteai.search.embedder import _STOP
from minuteai.search.index import IndexEntry, SearchIndex

RRF_K = 60
CANDIDATES = 50


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@dataclass
class Hit:
    position: int
    entry: IndexEntry
    score: float
    bm25_rank: int | None
    dense_rank: int | None


def matches(entry: IndexEntry, f: SearchFilters) -> bool:
    if f.meeting_ids and entry.meeting_id not in f.meeting_ids:
        return False
    if f.series_id and entry.series_id != f.series_id:
        return False
    if f.artifact_types and entry.artifact_type not in f.artifact_types:
        return False
    day = entry.occurred_at.date() if entry.occurred_at else None
    if f.date_from and (day is None or day < f.date_from):
        return False
    if f.date_to and (day is None or day > f.date_to):
        return False
    if f.participant:
        p = f.participant.strip().lower()
        if not any(p == x.lower() or p in x.lower().split() for x in entry.people):
            return False
    return True


class HybridRetriever:
    def __init__(self, index: SearchIndex) -> None:
        self.index = index
        self._bm25: BM25Okapi | None = None
        self._bm25_size = -1

    def _bm25_model(self) -> BM25Okapi | None:
        if self._bm25_size != len(self.index.entries) or self._bm25 is None:
            corpus = [tokenize(e.text) or ["_"] for e in self.index.entries]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._bm25_size = len(corpus)
        return self._bm25

    def retrieve(self, query: str, filters: SearchFilters | None = None, k: int = 8,
                 mode: str = "hybrid") -> list[Hit]:
        """mode: "hybrid" (RRF of both), or "bm25" / "dense" alone (for ablations)."""
        filters = filters or SearchFilters()
        with self.index.lock:
            entries = list(self.index.entries)
            if not entries:
                return []
            mask = np.array([matches(e, filters) for e in entries], dtype=bool)
            if not mask.any():
                return []

            bm25_ranks: dict[int, int] = {}
            model = self._bm25_model()
            q_tokens = tokenize(query)
            if model is not None and q_tokens:
                scores = np.asarray(model.get_scores(q_tokens))
                scores[~mask] = -np.inf
                order = [int(i) for i in np.argsort(-scores)[:CANDIDATES] if scores[i] > 0]
                bm25_ranks = {pos: r for r, pos in enumerate(order)}

            dense = self.index.dense_search(query, CANDIDATES, mask)
            dense_ranks = {pos: r for r, (pos, _) in enumerate(dense)}

        fused: dict[int, float] = {}
        sources = {"hybrid": (bm25_ranks, dense_ranks), "bm25": (bm25_ranks,),
                   "dense": (dense_ranks,)}[mode]
        for ranks in sources:
            for pos, r in ranks.items():
                fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + r + 1)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            Hit(position=pos, entry=entries[pos], score=round(score, 6),
                bm25_rank=bm25_ranks.get(pos), dense_rank=dense_ranks.get(pos))
            for pos, score in top
        ]
