"""Natural-language search over every meeting."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from minuteai.api.deps import get_services
from minuteai.api.processing import Services
from minuteai.api.schemas import SearchRequest, SearchResponse, SearchSource
from minuteai.extract.llm_client import LLMError

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, services: Services = Depends(get_services)) -> SearchResponse:
    if services.search is None:
        raise HTTPException(501, "search is not configured")
    try:
        result = services.search.answer(body.query, body.filters, body.top_k)
    except LLMError as e:
        raise HTTPException(502, f"answer generation failed: {e}") from e
    sources = [
        SearchSource(
            n=i + 1, meeting_id=h.entry.meeting_id, meeting_title=h.entry.meeting_title,
            occurred_at=h.entry.occurred_at, artifact_type=h.entry.artifact_type,
            artifact_id=h.entry.artifact_id, start_utt=h.entry.start_utt,
            end_utt=h.entry.end_utt, start_sec=h.entry.start_sec, end_sec=h.entry.end_sec,
            text=h.entry.text, score=h.score,
        )
        for i, h in enumerate(result.hits)
    ]
    return SearchResponse(
        query=body.query, answerable=result.answerable, answer=result.sentence.text,
        evidence=result.sentence.evidence, validated=result.sentence.validated,
        flags=result.flags, sources=sources,
    )
