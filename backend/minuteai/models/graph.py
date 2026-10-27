"""Cross-meeting accountability graph models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from minuteai.models.evidence import EvidenceSpan

CommitmentState = Literal[
    "PROPOSED", "COMMITTED", "FULFILLED", "SLIPPED", "SILENTLY_DROPPED", "REVERSED"
]
TERMINAL_STATES: frozenset[str] = frozenset({"FULFILLED", "SILENTLY_DROPPED", "REVERSED"})
ResolutionMethod = Literal[
    "self_introduction", "direct_address", "participant_list", "voice_embedding",
    "owner_name", "unresolved",
]


class PersonIdentity(BaseModel):
    person_id: str
    display_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    # meeting_id -> speaker label in that meeting
    speaker_labels: dict[str, str] = Field(default_factory=dict)
    method: ResolutionMethod = "unresolved"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def resolved(self) -> bool:
        return self.method != "unresolved"


class CommitmentMention(BaseModel):
    meeting_id: str
    item_id: str
    occurred_at: datetime
    description: str
    owner: str | None = None
    due_date: date | None = None
    due_raw: str | None = None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    similarity: float = 1.0  # similarity to the commitment it was linked into


class Transition(BaseModel):
    commitment_id: str
    from_state: CommitmentState | None
    to_state: CommitmentState
    meeting_id: str
    occurred_at: datetime
    reason: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class Commitment(BaseModel):
    commitment_id: str
    series_id: str
    description: str
    owner_id: str | None = None
    owner_name: str | None = None
    origin_meeting_id: str
    created_at: datetime
    due_date: date | None = None
    state: CommitmentState = "COMMITTED"
    mentions: list[CommitmentMention] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)
    # meeting_id -> state at the end of that meeting (timeline cells)
    timeline: dict[str, CommitmentState] = Field(default_factory=dict)


class DecisionRef(BaseModel):
    meeting_id: str
    decision_id: str
    statement: str
    occurred_at: datetime
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class DecisionReversal(BaseModel):
    reversal_id: str
    series_id: str
    original: DecisionRef
    reversing: DecisionRef
    similarity: float
    reason: str


class SeriesMeetingRef(BaseModel):
    meeting_id: str
    title: str
    occurred_at: datetime


class CommitmentGraph(BaseModel):
    series_id: str
    meetings: list[SeriesMeetingRef] = Field(default_factory=list)
    commitments: list[Commitment] = Field(default_factory=list)
    people: list[PersonIdentity] = Field(default_factory=list)
    reversals: list[DecisionReversal] = Field(default_factory=list)


class LedgerEntry(BaseModel):
    commitment: Commitment
    series_id: str


class PersonLedger(BaseModel):
    person_id: str
    display_name: str | None
    entries: list[LedgerEntry] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
