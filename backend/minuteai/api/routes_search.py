"""Natural-language search (wired in phase 8)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from minuteai.api.schemas import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    raise HTTPException(501, "search is not implemented yet")
