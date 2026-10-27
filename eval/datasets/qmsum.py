"""QMSum loader (query-based meeting summarization; used for retrieval eval).

    git clone https://github.com/Yale-LILY/QMSum eval/data/qmsum

Each meeting JSON has ``meeting_transcripts`` (speaker turns) and
``specific_query_list`` entries with ``relevant_text_span`` turn ranges, which
give gold retrieval targets for hit@k / MRR.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from eval.common import DATA_DIR
from minuteai.models.transcript import Transcript, Utterance

QMSUM_DIR = DATA_DIR / "qmsum"
SECONDS_PER_WORD = 0.4  # QMSum has no timings; synthesize monotone ones


@dataclass
class QMSumQuery:
    meeting_id: str
    query: str
    answer: str
    relevant_spans: list[tuple[int, int]]  # inclusive turn (= utt_id) ranges


@dataclass
class QMSumMeeting:
    meeting_id: str
    transcript: Transcript
    queries: list[QMSumQuery]
    general_queries: list[tuple[str, str]]


def to_transcript(meeting_id: str, turns: list[dict]) -> Transcript:
    utts, t = [], 0.0
    labels: dict[str, str] = {}
    for i, turn in enumerate(turns):
        speaker = turn.get("speaker", "unknown")
        label = labels.setdefault(speaker, f"SPEAKER_{len(labels):02d}")
        text = " ".join(turn.get("content", "").split())
        dur = max(0.5, len(text.split()) * SECONDS_PER_WORD)
        utts.append(Utterance(utt_id=i, speaker_label=label, speaker_name=speaker, text=text,
                              start=round(t, 2), end=round(t + dur, 2)))
        t += dur + 0.3
    return Transcript(meeting_id=meeting_id, utterances=utts, duration_sec=t,
                      backend="synthetic", model_version="qmsum-reference")


def load_file(path: Path) -> QMSumMeeting:
    data = json.loads(path.read_text())
    mid = f"qmsum-{path.stem}"
    queries = [
        QMSumQuery(
            meeting_id=mid, query=q["query"], answer=q.get("answer", ""),
            relevant_spans=[(int(a), int(b)) for a, b in q.get("relevant_text_span", [])],
        )
        for q in data.get("specific_query_list", [])
    ]
    general = [(q["query"], q.get("answer", "")) for q in data.get("general_query_list", [])]
    return QMSumMeeting(mid, to_transcript(mid, data.get("meeting_transcripts", [])), queries,
                        general)


def load(split: str = "test", domain: str = "ALL", n: int | None = None,
         root: Path = QMSUM_DIR) -> list[QMSumMeeting]:
    folder = root / "data" / domain / split
    files = sorted(folder.glob("*.json")) if folder.exists() else []
    return [load_file(f) for f in files[:n]]
