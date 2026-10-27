"""Request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from minuteai.models.evidence import EvidenceSpan
from minuteai.models.meeting import Meeting, MeetingStatus
from minuteai.models.search import SearchFilters


class CreateMeetingRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, description="original audio file name")
    content_type: str = "audio/mpeg"
    series_name: str | None = None
    occurred_at: datetime | None = None
    participants: list[str] = Field(default_factory=list)


class CreateMeetingResponse(BaseModel):
    meeting: Meeting
    upload_url: str
    audio_s3_key: str


class ProcessRequest(BaseModel):
    force: bool = False
    backend: Literal["whisper", "aws_transcribe"] | None = None


class ProcessResponse(BaseModel):
    meeting_id: str
    status: MeetingStatus


class MeetingList(BaseModel):
    meetings: list[Meeting]


class AudioUrl(BaseModel):
    url: str
    expires_in: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    s3: str
    dynamodb: str
    llm_model: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=8, ge=1, le=30)


class SearchSource(BaseModel):
    n: int
    meeting_id: str
    meeting_title: str
    occurred_at: datetime | None
    artifact_type: str
    artifact_id: str | None
    start_utt: int
    end_utt: int
    start_sec: float
    end_sec: float
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    answerable: bool
    answer: str
    evidence: list[EvidenceSpan]
    validated: bool
    flags: list[str]
    sources: list[SearchSource]


class SeriesSummary(BaseModel):
    series_id: str
    meetings: int
    last_occurred_at: datetime | None
