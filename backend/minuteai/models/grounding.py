"""Grounding validation results and the dropped-artifact record."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from minuteai.models.evidence import EvidenceSpan

CheckName = Literal[
    "existence", "quote_fidelity", "timestamp", "speaker", "entity", "non_empty"
]
FailReason = Literal[
    "UTT_NOT_FOUND",
    "QUOTE_NOT_FOUND",
    "TIMESTAMP_MISMATCH",
    "SPEAKER_MISMATCH",
    "ENTITY_NOT_SUPPORTED",
    "NO_EVIDENCE",
]
ArtifactType = Literal["abstract", "topic_summary", "decision", "action_item", "search_answer"]


class CheckFailure(BaseModel):
    check: CheckName
    reason: FailReason
    detail: str = ""


class ValidationResult(BaseModel):
    passed: bool
    failures: list[CheckFailure] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def reasons(self) -> set[str]:
        return {f.reason for f in self.failures}


class DroppedArtifact(BaseModel):
    meeting_id: str
    artifact_type: ArtifactType
    artifact_id: str
    text: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    failures: list[CheckFailure] = Field(default_factory=list)
    confidence: float = 0.0
    mode: str = "BALANCED"
    dropped_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
