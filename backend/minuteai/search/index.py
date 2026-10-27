"""Global vector index over utterance windows, decisions and action items.

FAISS ``IndexFlatIP`` over L2-normalized embeddings (inner product == cosine),
persisted to S3 at ``indexes/global.faiss`` with a JSON sidecar mapping each
vector position to (meeting_id, utt range, artifact type, artifact id).
Without faiss installed the same vectors are searched with numpy and stored
as ``indexes/global.npy``. Updates are incremental per meeting.
"""

from __future__ import annotations

import io
import json
import logging
import threading
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from minuteai.extract.speakers import name_speakers
from minuteai.models.artifacts import MeetingSummary
from minuteai.models.meeting import Meeting
from minuteai.models.transcript import Transcript
from minuteai.search.embedder import Embedder, get_embedder, utterance_windows
from minuteai.storage.s3_store import S3Store

log = logging.getLogger(__name__)

EntryType = Literal["utterances", "decision", "action_item"]
INDEX_KEY = "indexes/global.faiss"
NUMPY_KEY = "indexes/global.npy"
SIDECAR_KEY = "indexes/global.json"


class IndexEntry(BaseModel):
    meeting_id: str
    artifact_type: EntryType
    artifact_id: str | None = None
    start_utt: int
    end_utt: int
    start_sec: float
    end_sec: float
    text: str
    occurred_at: datetime | None = None
    series_id: str | None = None
    meeting_title: str = ""
    people: list[str] = Field(default_factory=list)  # speaker labels + names + owners


def _faiss():
    try:
        import faiss

        return faiss
    except ImportError:
        return None


def build_entries(
    meeting: Meeting, transcript: Transcript, summary: MeetingSummary | None
) -> list[IndexEntry]:
    if not any(u.speaker_name for u in transcript.utterances):
        transcript = name_speakers(transcript, meeting.participants or None)
    base = dict(meeting_id=meeting.meeting_id, occurred_at=meeting.occurred_at,
                series_id=meeting.series_id, meeting_title=meeting.title)
    entries: list[IndexEntry] = []
    for window in utterance_windows(transcript.utterances):
        people = sorted({p for u in window for p in (u.speaker_label, u.speaker_name) if p})
        entries.append(IndexEntry(
            **base, artifact_type="utterances", start_utt=window[0].utt_id,
            end_utt=window[-1].utt_id, start_sec=window[0].start, end_sec=window[-1].end,
            text=" ".join(f"{u.speaker}: {u.text}" for u in window), people=people,
        ))
    if summary:
        for d in summary.decisions:
            if not d.evidence:
                continue
            entries.append(IndexEntry(
                **base, artifact_type="decision", artifact_id=d.decision_id,
                start_utt=min(e.utt_id for e in d.evidence),
                end_utt=max(e.utt_id for e in d.evidence),
                start_sec=min(e.start_sec for e in d.evidence),
                end_sec=max(e.end_sec for e in d.evidence),
                text=f"Decision: {d.statement}" + (f" — {d.rationale}" if d.rationale else ""),
                people=sorted(set(d.participants)),
            ))
        for a in summary.action_items:
            if not a.evidence:
                continue
            owner = [p for p in (a.owner, a.owner_speaker_label) if p]
            entries.append(IndexEntry(
                **base, artifact_type="action_item", artifact_id=a.item_id,
                start_utt=min(e.utt_id for e in a.evidence),
                end_utt=max(e.utt_id for e in a.evidence),
                start_sec=min(e.start_sec for e in a.evidence),
                end_sec=max(e.end_sec for e in a.evidence),
                text=f"Action item: {a.description}"
                + (f" (owner: {a.owner})" if a.owner else " (no clear owner)")
                + (f", due {a.due_raw}" if a.due_raw else ""),
                people=owner,
            ))
    return entries


class SearchIndex:
    def __init__(self, embedder: Embedder | None = None, s3: S3Store | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self.s3 = s3
        self.entries: list[IndexEntry] = []
        self.vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._faiss_index = None
        self.lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.entries)

    # -- mutation ------------------------------------------------------------
    def add_meeting(
        self, meeting: Meeting, transcript: Transcript, summary: MeetingSummary | None
    ) -> int:
        new = build_entries(meeting, transcript, summary)
        vecs = self.embedder.embed([e.text for e in new]) if new else \
            np.zeros((0, self.embedder.dim), dtype=np.float32)
        with self.lock:
            keep = [i for i, e in enumerate(self.entries) if e.meeting_id != meeting.meeting_id]
            self.entries = [self.entries[i] for i in keep] + new
            self.vectors = np.vstack([self.vectors[keep], vecs]).astype(np.float32)
            self._faiss_index = None
        return len(new)

    def remove_meeting(self, meeting_id: str) -> None:
        with self.lock:
            keep = [i for i, e in enumerate(self.entries) if e.meeting_id != meeting_id]
            self.entries = [self.entries[i] for i in keep]
            self.vectors = self.vectors[keep]
            self._faiss_index = None

    # -- query ---------------------------------------------------------------
    def _index(self):
        faiss = _faiss()
        if faiss is None:
            return None
        if self._faiss_index is None:
            idx = faiss.IndexFlatIP(self.embedder.dim)
            if len(self.vectors):
                idx.add(self.vectors)
            self._faiss_index = idx
        return self._faiss_index

    def dense_search(self, query: str, k: int, mask: np.ndarray | None = None
                     ) -> list[tuple[int, float]]:
        with self.lock:
            if not self.entries:
                return []
            q = self.embedder.embed([query])
            if mask is not None and not mask.all():
                # Filtered search: exact scoring over the allowed subset.
                allowed = np.flatnonzero(mask)
                scores = self.vectors[allowed] @ q[0]
                order = np.argsort(-scores)[:k]
                return [(int(allowed[i]), float(scores[i])) for i in order]
            idx = self._index()
            if idx is not None:
                scores, ids = idx.search(q, min(k, len(self.entries)))
                return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=False)
                        if i >= 0]
            scores = self.vectors @ q[0]
            order = np.argsort(-scores)[:k]
            return [(int(i), float(scores[i])) for i in order]

    # -- persistence -----------------------------------------------------------
    def save(self) -> None:
        if self.s3 is None:
            return
        with self.lock:
            sidecar = {
                "embedder": self.embedder.name, "dim": self.embedder.dim,
                "entries": [e.model_dump(mode="json") for e in self.entries],
            }
            faiss = _faiss()
            if faiss is not None:
                blob = faiss.serialize_index(self._index()).tobytes()
                self.s3.put_bytes(INDEX_KEY, blob)
            else:
                buf = io.BytesIO()
                np.save(buf, self.vectors)
                self.s3.put_bytes(NUMPY_KEY, buf.getvalue())
            self.s3.put_text(SIDECAR_KEY, json.dumps(sidecar))

    def load(self) -> bool:
        """Load from S3. Returns False if missing or built with another embedder."""
        if self.s3 is None or not self.s3.exists(SIDECAR_KEY):
            return False
        sidecar = json.loads(self.s3.get_text(SIDECAR_KEY))
        if sidecar.get("embedder") != self.embedder.name or sidecar.get("dim") != self.embedder.dim:
            log.info("index built with %s, current embedder %s — needs rebuild",
                     sidecar.get("embedder"), self.embedder.name)
            return False
        faiss = _faiss()
        if faiss is not None and self.s3.exists(INDEX_KEY):
            blob = np.frombuffer(self.s3.get_bytes(INDEX_KEY), dtype="uint8")
            idx = faiss.deserialize_index(blob)
            vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
            if idx.ntotal:
                vectors = idx.reconstruct_n(0, idx.ntotal)
        elif self.s3.exists(NUMPY_KEY):
            vectors = np.load(io.BytesIO(self.s3.get_bytes(NUMPY_KEY)))
        else:
            return False
        with self.lock:
            self.entries = [IndexEntry.model_validate(e) for e in sidecar["entries"]]
            self.vectors = np.asarray(vectors, dtype=np.float32)
            self._faiss_index = None
        return len(self.entries) == len(self.vectors)
