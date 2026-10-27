"""Meeting metadata and processing status."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

MeetingStatus = Literal["uploaded", "transcribing", "extracting", "ready", "failed"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class Meeting(BaseModel):
    meeting_id: str
    title: str
    series_id: str | None = None
    occurred_at: datetime = Field(default_factory=utcnow)
    duration_sec: float = 0.0
    audio_s3_key: str | None = None
    participants: list[str] = Field(default_factory=list)
    status: MeetingStatus = "uploaded"
    created_at: datetime = Field(default_factory=utcnow)
    error: str | None = None
