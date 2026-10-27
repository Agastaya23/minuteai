"""Response schemas the LLM fills in. Content only — ids, timestamps and speaker
labels on the final artifacts are attached deterministically by the extractor.

Kept deliberately flat (no numeric bounds, no defaults) so they translate
cleanly into provider structured-output schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LLMEvidence(BaseModel):
    utt_id: int = Field(description="id in square brackets of a supporting utterance")
    quote: str = Field(description="short verbatim excerpt (<= 25 words) from that utterance")


class LLMSentence(BaseModel):
    text: str
    evidence: list[LLMEvidence]


class LLMSummaryResponse(BaseModel):
    sentences: list[LLMSentence]


class LLMDecision(BaseModel):
    statement: str
    rationale: str | None
    participants: list[str] = Field(description="speaker labels, e.g. SPEAKER_01")
    status: Literal["proposed", "agreed", "deferred", "reversed"]
    confidence: float = Field(description="0.0-1.0")
    evidence: list[LLMEvidence]


class LLMDecisionsResponse(BaseModel):
    decisions: list[LLMDecision]


class LLMActionItem(BaseModel):
    description: str
    owner: str | None = Field(description="person's name, or null if nobody is clearly assigned")
    owner_speaker_label: str | None = Field(description="speaker label of the owner, or null")
    owner_resolution_method: Literal[
        "explicit_name", "self_commitment", "assigned_by_other", "unresolved"
    ]
    owner_confidence: float = Field(description="0.0-1.0")
    due_raw: str | None = Field(description="the original deadline phrase, verbatim, or null")
    confidence: float = Field(description="0.0-1.0")
    evidence: list[LLMEvidence]


class LLMActionItemsResponse(BaseModel):
    action_items: list[LLMActionItem]


class LLMTopicTitle(BaseModel):
    segment: int
    title: str


class LLMTopicTitlesResponse(BaseModel):
    topics: list[LLMTopicTitle]


class LLMAnswer(BaseModel):
    answerable: bool
    answer: str
    evidence: list[LLMEvidence] = Field(description="utt_id is the [n] source number")


class LLMBaselineResponse(BaseModel):
    """Ablation config A: no evidence requested."""

    summary: list[str]
    decisions: list[str]
    action_items: list[str]


class LLMCommitmentStatus(BaseModel):
    status: Literal["fulfilled", "slipped", "reversed", "in_progress", "not_mentioned"]
    new_due_raw: str | None
    evidence: list[LLMEvidence]
