"""Transcript chunking and topic segmentation.

Boundaries are chosen where the content shifts: cosine distance between the
embeddings of the utterance windows immediately left and right of each gap.
Token-count splits are the fallback. Real utt_ids are always preserved.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from minuteai.models.transcript import Transcript, Utterance
from minuteai.search.embedder import Embedder, get_embedder

BOUNDARY_WINDOW = 4


class TranscriptChunk(BaseModel):
    chunk_id: int
    start_utt: int
    end_utt: int
    utterances: list[Utterance]


def fmt_ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else \
        f"{sec // 60:02d}:{sec % 60:02d}"


def render_utterance(u: Utterance) -> str:
    who = f"{u.speaker_label} ({u.speaker_name})" if u.speaker_name else u.speaker_label
    return f"[{u.utt_id}] {who} ({fmt_ts(u.start)}): {u.text}"


def render_utterances(utts: list[Utterance]) -> str:
    return "\n".join(render_utterance(u) for u in utts)


def estimate_tokens(u: Utterance) -> int:
    # ~4 chars/token for English, plus the "[id] SPEAKER_XX (mm:ss): " prefix.
    return len(u.text) // 4 + 12


def boundary_scores(
    utterances: list[Utterance], embedder: Embedder | None = None, window: int = BOUNDARY_WINDOW
) -> np.ndarray:
    """scores[i] = topic-shift strength of the gap *before* utterances[i] (scores[0] = 0)."""
    n = len(utterances)
    scores = np.zeros(n, dtype=np.float32)
    if n < 2 * window:
        return scores
    embedder = embedder or get_embedder()
    texts = [u.text for u in utterances]
    vecs = embedder.embed(texts)
    # prefix sums give O(n) window means
    csum = np.vstack([np.zeros((1, vecs.shape[1]), dtype=np.float32), np.cumsum(vecs, axis=0)])
    for i in range(window, n - window + 1):
        left = csum[i] - csum[i - window]
        right = csum[i + window] - csum[i]
        denom = (np.linalg.norm(left) * np.linalg.norm(right)) or 1.0
        scores[i] = 1.0 - float(np.dot(left, right) / denom)
    return scores


def chunk_transcript(
    transcript: Transcript,
    max_tokens: int = 6000,
    overlap_utts: int = 3,
    embedder: Embedder | None = None,
) -> list[TranscriptChunk]:
    utts = transcript.utterances
    if not utts:
        return []
    tokens = [estimate_tokens(u) for u in utts]
    if sum(tokens) <= max_tokens:
        return [TranscriptChunk(chunk_id=0, start_utt=utts[0].utt_id, end_utt=utts[-1].utt_id,
                                utterances=list(utts))]
    scores = boundary_scores(utts, embedder)
    chunks: list[TranscriptChunk] = []
    start = 0
    while start < len(utts):
        # furthest index we can include without exceeding the budget
        budget, limit = 0, start
        while limit < len(utts) and budget + tokens[limit] <= max_tokens:
            budget += tokens[limit]
            limit += 1
        limit = max(limit, start + 1)
        if limit >= len(utts):
            end = len(utts)
        else:
            # prefer the strongest topic boundary in the back half of the window
            lo = start + max(overlap_utts + 1, (limit - start) // 2)
            window = scores[lo:limit]
            end = lo + int(np.argmax(window)) if window.size and window.max() > 0 else limit
        chunk_utts = utts[start:end]
        chunks.append(TranscriptChunk(chunk_id=len(chunks), start_utt=chunk_utts[0].utt_id,
                                      end_utt=chunk_utts[-1].utt_id, utterances=chunk_utts))
        if end >= len(utts):
            break
        start = max(end - overlap_utts, start + 1)
    return chunks


def segment_topics(
    transcript: Transcript,
    min_utts: int = 6,
    max_topics: int = 12,
    embedder: Embedder | None = None,
) -> list[tuple[int, int]]:
    """Deterministic topic segmentation → [(start_utt, end_utt)], inclusive, contiguous."""
    utts = transcript.utterances
    n = len(utts)
    if n == 0:
        return []
    if n < 2 * min_utts:
        return [(utts[0].utt_id, utts[-1].utt_id)]
    scores = boundary_scores(utts, embedder, window=min(BOUNDARY_WINDOW, min_utts))
    threshold = float(scores[scores > 0].mean() + 0.5 * scores[scores > 0].std()) \
        if (scores > 0).any() else 1.0
    cuts: list[int] = []
    for i in np.argsort(-scores):
        i = int(i)
        if scores[i] <= threshold or len(cuts) >= max_topics - 1:
            break
        if i < min_utts or n - i < min_utts or any(abs(i - c) < min_utts for c in cuts):
            continue
        cuts.append(i)
    bounds = [0, *sorted(cuts), n]
    return [(utts[a].utt_id, utts[b - 1].utt_id) for a, b in zip(bounds, bounds[1:], strict=False)]
