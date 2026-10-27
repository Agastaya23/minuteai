"""EvidenceSpan — the backbone of the system.

Every extracted artifact (summary sentence, decision, action item, search
answer) carries a list of these. A span points at one real utterance and
quotes the exact text from it that supports the claim. The grounding
validator checks each field against the transcript.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_QUOTE_CHARS = 200


class EvidenceSpan(BaseModel):
    utt_id: int = Field(ge=0)
    start_sec: float
    end_sec: float
    speaker_label: str
    quote: str = Field(max_length=MAX_QUOTE_CHARS)
    meeting_id: str | None = None  # set for cross-meeting evidence (search, graph)

    @field_validator("quote", mode="before")
    @classmethod
    def _truncate(cls, v: str) -> str:
        v = (v or "").strip()
        return v[:MAX_QUOTE_CHARS]
