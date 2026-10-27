"""Extracted artifacts. Every one carries `evidence: list[EvidenceSpan]`."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from minuteai.models.evidence import EvidenceSpan

DecisionStatus = Literal["proposed", "agreed", "deferred", "reversed"]
OwnerResolution = Literal["explicit_name", "self_commitment", "assigned_by_other", "unresolved"]


class GroundedSentence(BaseModel):
    text: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    validated: bool = False
    flags: list[str] = Field(default_factory=list)  # non-fatal grounding failures


class Decision(BaseModel):
    decision_id: str
    statement: str
    rationale: str | None = None
    participants: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: DecisionStatus = "agreed"
    validated: bool = False
    flags: list[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    item_id: str
    description: str
    owner: str | None = None
    owner_speaker_label: str | None = None
    owner_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    owner_resolution_method: OwnerResolution = "unresolved"
    due_date: date | None = None
    due_raw: str | None = None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    validated: bool = False
    flags: list[str] = Field(default_factory=list)


class Topic(BaseModel):
    topic_id: str
    title: str
    start_utt: int
    end_utt: int
    summary: list[GroundedSentence] = Field(default_factory=list)


class MeetingSummary(BaseModel):
    meeting_id: str
    abstract: list[GroundedSentence] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_version: str = ""
    grounding_mode: str | None = None
