"""Search filters shared by the retrieval layer and the API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel

SearchArtifactType = Literal["utterances", "decision", "action_item"]


class SearchFilters(BaseModel):
    date_from: date | None = None
    date_to: date | None = None
    series_id: str | None = None
    participant: str | None = None
    artifact_types: list[SearchArtifactType] | None = None
    meeting_ids: list[str] | None = None
