"""Chunked, evidence-carrying extraction.

The LLM supplies content (text, owner names, quotes, cited utt_ids). Everything
structural — ids, timestamps, speaker labels on evidence, chunk and topic
boundaries, cross-chunk merging, due-date resolution — is deterministic here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import TypeVar

import numpy as np
from rapidfuzz import fuzz

from minuteai.extract import prompts
from minuteai.extract.chunker import (
    TranscriptChunk,
    chunk_transcript,
    render_utterances,
    segment_topics,
)
from minuteai.extract.dates import resolve_relative_date
from minuteai.extract.llm_client import LLMClient, LLMError
from minuteai.extract.schemas import (
    LLMActionItemsResponse,
    LLMBaselineResponse,
    LLMDecisionsResponse,
    LLMEvidence,
    LLMSummaryResponse,
    LLMTopicTitlesResponse,
)
from minuteai.extract.speakers import name_speakers
from minuteai.models.artifacts import (
    ActionItem,
    Decision,
    GroundedSentence,
    MeetingSummary,
    Topic,
)
from minuteai.models.evidence import EvidenceSpan
from minuteai.models.transcript import Transcript, Utterance
from minuteai.search.embedder import Embedder, get_embedder

log = logging.getLogger(__name__)

A = TypeVar("A")
DEDUP_THRESHOLD = 0.88
TOPIC_SAMPLE_UTTS = 30


def build_evidence(
    items: Sequence[LLMEvidence], lookup: Callable[[int], Utterance | None],
    meeting_id: str | None = None,
) -> list[EvidenceSpan]:
    """Attach real timestamps/speakers to LLM-cited utt_ids.

    Unknown utt_ids are kept (with zeroed timing and speaker UNKNOWN) so the
    grounding validator can reject them and the rejection can be logged.
    """
    spans: list[EvidenceSpan] = []
    seen: set[tuple[int, str]] = set()
    for ev in items:
        if ev.utt_id < 0:
            log.warning("dropping evidence with negative utt_id %s", ev.utt_id)
            continue
        key = (ev.utt_id, ev.quote.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        u = lookup(ev.utt_id)
        spans.append(
            EvidenceSpan(
                utt_id=ev.utt_id,
                start_sec=u.start if u else 0.0,
                end_sec=u.end if u else 0.0,
                speaker_label=u.speaker_label if u else "UNKNOWN",
                quote=ev.quote,
                meeting_id=meeting_id,
            )
        )
    return spans


def _first_utt(evidence: list[EvidenceSpan]) -> int:
    return min((e.utt_id for e in evidence), default=10**9)


def _merge_evidence(a: list[EvidenceSpan], b: list[EvidenceSpan]) -> list[EvidenceSpan]:
    seen = {(e.utt_id, e.quote) for e in a}
    return sorted([*a, *(e for e in b if (e.utt_id, e.quote) not in seen)],
                  key=lambda e: e.utt_id)


def dedup(
    items: list[A],
    text_of: Callable[[A], str],
    merge: Callable[[A, A], A],
    embedder: Embedder,
    threshold: float = DEDUP_THRESHOLD,
    compatible: Callable[[A, A], bool] = lambda a, b: True,
) -> list[A]:
    """Greedy near-duplicate merge (overlapping chunks re-extract the same item)."""
    if len(items) < 2:
        return list(items)
    vecs = embedder.embed([text_of(i) for i in items])
    kept: list[int] = []
    merged: dict[int, A] = {}
    for idx, item in enumerate(items):
        target = None
        for k in kept:
            sim = max(float(np.dot(vecs[idx], vecs[k])),
                      fuzz.token_set_ratio(text_of(item), text_of(merged[k])) / 100)
            if sim >= threshold and compatible(merged[k], item):
                target = k
                break
        if target is None:
            kept.append(idx)
            merged[idx] = item
        else:
            merged[target] = merge(merged[target], item)
    return [merged[k] for k in kept]


class Extractor:
    def __init__(
        self,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        max_tokens: int = 6000,
        overlap_utts: int = 3,
        max_workers: int = 4,
    ) -> None:
        self.llm = llm or LLMClient()
        self.embedder = embedder or get_embedder()
        self.max_tokens = max_tokens
        self.overlap_utts = overlap_utts
        self.max_workers = max_workers

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _lookup(chunk: TranscriptChunk, transcript: Transcript | None):
        if transcript is not None:
            return transcript.get
        by_id = {u.utt_id: u for u in chunk.utterances}
        return by_id.get

    def _call(self, template: str, model, purpose: str, meeting_id: str | None, **kw):
        return self.llm.complete_structured(
            prompts.render(template, **kw), model, purpose=purpose, meeting_id=meeting_id
        )

    # -- per-chunk extraction -----------------------------------------------
    def extract_summary(
        self, chunk: TranscriptChunk, transcript: Transcript | None = None
    ) -> list[GroundedSentence]:
        mid = transcript.meeting_id if transcript else None
        resp = self._call(prompts.SUMMARY_PROMPT, LLMSummaryResponse, "summary", mid,
                          transcript=render_utterances(chunk.utterances))
        lookup = self._lookup(chunk, transcript)
        return [
            GroundedSentence(text=s.text.strip(), evidence=build_evidence(s.evidence, lookup))
            for s in resp.sentences if s.text.strip()
        ]

    def extract_decisions(
        self, chunk: TranscriptChunk, transcript: Transcript | None = None
    ) -> list[Decision]:
        mid = transcript.meeting_id if transcript else None
        resp = self._call(prompts.DECISIONS_PROMPT, LLMDecisionsResponse, "decisions", mid,
                          transcript=render_utterances(chunk.utterances))
        lookup = self._lookup(chunk, transcript)
        return [
            Decision(
                decision_id=f"c{chunk.chunk_id}-d{i}",
                statement=d.statement.strip(),
                rationale=d.rationale,
                participants=sorted(set(d.participants)),
                evidence=build_evidence(d.evidence, lookup),
                confidence=min(1.0, max(0.0, d.confidence)),
                status=d.status,
            )
            for i, d in enumerate(resp.decisions) if d.statement.strip()
        ]

    def extract_action_items(
        self,
        chunk: TranscriptChunk,
        transcript: Transcript | None = None,
        reference_date: date | None = None,
    ) -> list[ActionItem]:
        mid = transcript.meeting_id if transcript else None
        resp = self._call(prompts.ACTION_ITEMS_PROMPT, LLMActionItemsResponse, "action_items",
                          mid, transcript=render_utterances(chunk.utterances))
        lookup = self._lookup(chunk, transcript)
        names = {u.speaker_label: u.speaker_name for u in chunk.utterances if u.speaker_name}
        ref = reference_date or date.today()
        items: list[ActionItem] = []
        for i, a in enumerate(resp.action_items):
            if not a.description.strip():
                continue
            method = a.owner_resolution_method
            label = a.owner_speaker_label
            label = label.strip() if label and label.strip().startswith("SPEAKER_") else None
            owner = names.get(label) if label else None
            owner = owner or (a.owner.strip() if a.owner and a.owner.strip() else None) or label
            if method == "unresolved" or owner is None:
                owner, label, method, owner_conf = None, None, "unresolved", 0.0
            else:
                owner_conf = min(1.0, max(0.0, a.owner_confidence))
            items.append(
                ActionItem(
                    item_id=f"c{chunk.chunk_id}-a{i}",
                    description=a.description.strip(),
                    owner=owner,
                    owner_speaker_label=label,
                    owner_confidence=owner_conf,
                    owner_resolution_method=method,
                    due_raw=a.due_raw,
                    due_date=resolve_relative_date(a.due_raw, ref),
                    evidence=build_evidence(a.evidence, lookup),
                    confidence=min(1.0, max(0.0, a.confidence)),
                )
            )
        return items

    # -- topics ----------------------------------------------------------------
    def extract_topics(
        self, transcript: Transcript, sentences: list[GroundedSentence] | None = None
    ) -> list[Topic]:
        segments = segment_topics(transcript, embedder=self.embedder)
        if not segments:
            return []
        blocks = []
        for n, (a, b) in enumerate(segments):
            utts = transcript.window(a, b)
            if len(utts) > TOPIC_SAMPLE_UTTS:
                picks = np.linspace(0, len(utts) - 1, TOPIC_SAMPLE_UTTS).astype(int)
                utts = [utts[i] for i in sorted(set(picks))]
            blocks.append(f"Segment {n}:\n{render_utterances(utts)}")
        titles: dict[int, str] = {}
        try:
            resp = self._call(prompts.TOPIC_TITLES_PROMPT, LLMTopicTitlesResponse, "topics",
                              transcript.meeting_id, segments="\n\n".join(blocks))
            titles = {t.segment: t.title.strip() for t in resp.topics if t.title.strip()}
        except LLMError as e:
            log.warning("topic titling failed, using placeholders: %s", e)
        topics = []
        for n, (a, b) in enumerate(segments):
            summary = [
                s for s in (sentences or [])
                if s.evidence and a <= _first_utt(s.evidence) <= b
            ]
            topics.append(Topic(topic_id=f"t{n}", title=titles.get(n, f"Topic {n + 1}"),
                                start_utt=a, end_utt=b, summary=summary))
        return topics

    # -- orchestration -------------------------------------------------------
    def extract_meeting(
        self,
        transcript: Transcript,
        occurred_at: datetime | None = None,
        participants: list[str] | None = None,
    ) -> MeetingSummary:
        transcript = name_speakers(transcript, participants)
        chunks = chunk_transcript(transcript, self.max_tokens, self.overlap_utts, self.embedder)
        ref = (occurred_at or datetime.now()).date()

        jobs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for c in chunks:
                jobs.append((
                    pool.submit(self.extract_summary, c, transcript),
                    pool.submit(self.extract_decisions, c, transcript),
                    pool.submit(self.extract_action_items, c, transcript, ref),
                ))
            per_chunk = [(s.result(), d.result(), a.result()) for s, d, a in jobs]

        sentences = [s for chunk in per_chunk for s in chunk[0]]
        decisions = [d for chunk in per_chunk for d in chunk[1]]
        actions = [a for chunk in per_chunk for a in chunk[2]]
        sentences, decisions, actions = self.merge(sentences, decisions, actions)

        names = sorted({u.speaker for u in transcript.utterances})
        return MeetingSummary(
            meeting_id=transcript.meeting_id,
            abstract=sentences,
            topics=self.extract_topics(transcript, sentences),
            decisions=decisions,
            action_items=actions,
            participants=names,
            model_version=self.llm.model,
        )

    def merge(
        self,
        sentences: list[GroundedSentence],
        decisions: list[Decision],
        actions: list[ActionItem],
    ) -> tuple[list[GroundedSentence], list[Decision], list[ActionItem]]:
        by_pos = lambda x: _first_utt(x.evidence)  # noqa: E731
        sentences = dedup(
            sorted(sentences, key=by_pos), lambda s: s.text,
            lambda a, b: a.model_copy(update={"evidence": _merge_evidence(a.evidence, b.evidence)}),
            self.embedder,
        )
        decisions = dedup(
            sorted(decisions, key=by_pos), lambda d: d.statement,
            lambda a, b: a.model_copy(update={
                "evidence": _merge_evidence(a.evidence, b.evidence),
                "participants": sorted(set(a.participants) | set(b.participants)),
                "confidence": max(a.confidence, b.confidence),
            }),
            self.embedder,
        )
        actions = dedup(
            sorted(actions, key=by_pos), lambda a: a.description,
            _merge_actions, self.embedder,
            compatible=lambda a, b: a.owner is None or b.owner is None or a.owner == b.owner,
        )
        decisions = [d.model_copy(update={"decision_id": f"d{i}"}) for i, d in enumerate(decisions)]
        actions = [a.model_copy(update={"item_id": f"a{i}"}) for i, a in enumerate(actions)]
        return sentences, decisions, actions

    # -- ablation baseline (config A) ---------------------------------------
    def extract_meeting_baseline(self, transcript: Transcript) -> MeetingSummary:
        """Whole transcript in one prompt, no chunking, no evidence requested."""
        transcript = name_speakers(transcript)
        resp = self.llm.complete_structured(
            prompts.render(prompts.BASELINE_SINGLE_PROMPT,
                           transcript=render_utterances(transcript.utterances)),
            LLMBaselineResponse, purpose="baseline", meeting_id=transcript.meeting_id,
        )
        return MeetingSummary(
            meeting_id=transcript.meeting_id,
            abstract=[GroundedSentence(text=t) for t in resp.summary],
            decisions=[Decision(decision_id=f"d{i}", statement=t)
                       for i, t in enumerate(resp.decisions)],
            action_items=[ActionItem(item_id=f"a{i}", description=t)
                          for i, t in enumerate(resp.action_items)],
            participants=sorted({u.speaker for u in transcript.utterances}),
            model_version=f"{self.llm.model}/baseline",
        )


def _merge_actions(a: ActionItem, b: ActionItem) -> ActionItem:
    owner_src = a if a.owner or not b.owner else b
    return a.model_copy(update={
        "evidence": _merge_evidence(a.evidence, b.evidence),
        "owner": owner_src.owner,
        "owner_speaker_label": owner_src.owner_speaker_label,
        "owner_confidence": owner_src.owner_confidence,
        "owner_resolution_method": owner_src.owner_resolution_method,
        "due_raw": a.due_raw or b.due_raw,
        "due_date": a.due_date or b.due_date,
        "confidence": max(a.confidence, b.confidence),
    })


_default: Extractor | None = None


def _extractor() -> Extractor:
    global _default
    if _default is None:
        _default = Extractor()
    return _default


def extract_topics(transcript: Transcript) -> list[Topic]:
    return _extractor().extract_topics(transcript)


def extract_summary(chunk: TranscriptChunk) -> list[GroundedSentence]:
    return _extractor().extract_summary(chunk)


def extract_decisions(chunk: TranscriptChunk) -> list[Decision]:
    return _extractor().extract_decisions(chunk)


def extract_action_items(chunk: TranscriptChunk) -> list[ActionItem]:
    return _extractor().extract_action_items(chunk)


def extract_meeting(transcript: Transcript) -> MeetingSummary:
    return _extractor().extract_meeting(transcript)
