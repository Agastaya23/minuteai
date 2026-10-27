"""GroundingFilter — apply the validator across a MeetingSummary.

Modes (the three points of the precision/recall curve):
  STRICT      any failed check drops the artifact
  BALANCED    drops on existence, quote fidelity, non-empty; flags the rest (default)
  PERMISSIVE  flags only, drops nothing

Every dropped artifact is appended to data/dropped_artifacts.jsonl with its
text, evidence and failure reasons — the taxonomy of grounding failures is a
research finding in its own right.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import date
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from minuteai.config import get_settings
from minuteai.ground.validator import GroundingValidator, artifact_text
from minuteai.models.artifacts import MeetingSummary
from minuteai.models.grounding import ArtifactType, DroppedArtifact, ValidationResult
from minuteai.models.transcript import Transcript


class Strictness(StrEnum):
    STRICT = "STRICT"
    BALANCED = "BALANCED"
    PERMISSIVE = "PERMISSIVE"


DROP_REASONS: dict[Strictness, set[str] | None] = {
    Strictness.STRICT: None,  # None = every reason drops
    Strictness.BALANCED: {"UTT_NOT_FOUND", "QUOTE_NOT_FOUND", "NO_EVIDENCE"},
    Strictness.PERMISSIVE: set(),
}

_log_lock = threading.Lock()


class FilterReport(BaseModel):
    mode: str
    total: int = 0
    kept: int = 0
    dropped: int = 0
    flagged: int = 0
    passed: int = 0
    by_type: dict[str, dict[str, int]] = Field(default_factory=dict)
    reasons: dict[str, int] = Field(default_factory=dict)


def should_drop(result: ValidationResult, mode: Strictness) -> bool:
    if result.passed:
        return False
    drop = DROP_REASONS[mode]
    return True if drop is None else bool(result.reasons & drop)


class GroundingFilter:
    def __init__(
        self,
        transcript: Transcript | dict[str, Transcript],
        mode: Strictness | str | None = None,
        log_path: Path | None = None,
        reference_date: date | None = None,
        validator: GroundingValidator | None = None,
    ) -> None:
        self.mode = Strictness(mode or get_settings().grounding_mode)
        self.validator = validator or GroundingValidator(transcript, reference_date=reference_date)
        self.log_path = log_path or get_settings().data_path("dropped_artifacts.jsonl")
        self.report = FilterReport(mode=self.mode)

    def _judge(self, artifact, kind: ArtifactType, artifact_id: str, meeting_id: str, dropped,
               cache: dict[int, tuple[bool, object]]):
        """Returns (keep, annotated artifact)."""
        if id(artifact) in cache:
            return cache[id(artifact)]
        result = self.validator.validate(artifact)
        r = self.report
        r.total += 1
        stats = r.by_type.setdefault(kind, {"total": 0, "kept": 0, "dropped": 0, "flagged": 0})
        stats["total"] += 1
        for f in result.failures:
            r.reasons[f.reason] = r.reasons.get(f.reason, 0) + 1
        if should_drop(result, self.mode):
            r.dropped += 1
            stats["dropped"] += 1
            dropped.append(DroppedArtifact(
                meeting_id=meeting_id, artifact_type=kind, artifact_id=artifact_id,
                text=artifact_text(artifact), evidence=list(artifact.evidence),
                failures=result.failures, confidence=result.confidence, mode=self.mode,
            ))
            out = (False, artifact)
        else:
            r.kept += 1
            stats["kept"] += 1
            if result.passed:
                r.passed += 1
            else:
                r.flagged += 1
                stats["flagged"] += 1
            flags = sorted(result.reasons)
            update = {"validated": result.passed, "flags": flags}
            if hasattr(artifact, "confidence"):
                # confidence is capped by how well the evidence holds up
                update["confidence"] = round(min(artifact.confidence, result.confidence), 4)
            out = (True, artifact.model_copy(update=update))
        cache[id(artifact)] = out
        return out

    def apply(self, summary: MeetingSummary) -> tuple[MeetingSummary, list[DroppedArtifact]]:
        self.report = FilterReport(mode=self.mode)
        dropped: list[DroppedArtifact] = []
        mid = summary.meeting_id
        cache: dict[int, tuple[bool, object]] = {}
        # Topic summaries are typically the same sentence objects as the abstract;
        # judge by text so a sentence is validated (and logged) once.
        by_text: dict[str, tuple[bool, object]] = {}

        def sentence(s, kind, sid):
            if s.text in by_text:
                return by_text[s.text]
            res = self._judge(s, kind, sid, mid, dropped, cache)
            by_text[s.text] = res
            return res

        abstract = [a for i, s in enumerate(summary.abstract)
                    for keep, a in [sentence(s, "abstract", f"s{i}")] if keep]
        topics = []
        for t in summary.topics:
            kept = [a for j, s in enumerate(t.summary)
                    for keep, a in [sentence(s, "topic_summary", f"{t.topic_id}-s{j}")] if keep]
            topics.append(t.model_copy(update={"summary": kept}))
        decisions = [a for d in summary.decisions
                     for keep, a in [self._judge(d, "decision", d.decision_id, mid, dropped, cache)]
                     if keep]
        actions = [a for item in summary.action_items
                   for keep, a in [self._judge(item, "action_item", item.item_id, mid, dropped,
                                               cache)] if keep]
        kept = summary.model_copy(update={
            "abstract": abstract, "topics": topics, "decisions": decisions,
            "action_items": actions, "grounding_mode": str(self.mode),
        })
        self._write_log(dropped)
        return kept, dropped

    def _write_log(self, dropped: list[DroppedArtifact]) -> None:
        if not dropped:
            return
        with _log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                for d in dropped:
                    f.write(json.dumps(d.model_dump(mode="json")) + "\n")


def reason_breakdown(dropped: list[DroppedArtifact]) -> Counter:
    return Counter(f.reason for d in dropped for f in d.failures)
