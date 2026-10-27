"""Cross-meeting accountability graph endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from minuteai.api.deps import get_services
from minuteai.api.processing import Services
from minuteai.api.schemas import SeriesSummary
from minuteai.models.graph import CommitmentGraph, DecisionReversal, PersonLedger

router = APIRouter(tags=["graph"])


def _graph_service(services: Services):
    if services.graph is None:
        raise HTTPException(501, "accountability graph is not configured")
    return services.graph


@router.get("/series", response_model=list[SeriesSummary])
def list_series(services: Services = Depends(get_services)) -> list[SeriesSummary]:
    by_series: dict[str, list] = {}
    for m in services.repo.list_meetings():
        if m.series_id:
            by_series.setdefault(m.series_id, []).append(m)
    return [
        SeriesSummary(series_id=sid, meetings=len(ms),
                      last_occurred_at=max(m.occurred_at for m in ms))
        for sid, ms in sorted(by_series.items(), key=lambda kv: -max(
            m.occurred_at for m in kv[1]).timestamp())
    ]


@router.get("/series/{series_id}/commitments", response_model=CommitmentGraph)
def get_commitments(series_id: str, services: Services = Depends(get_services)):
    if not services.repo.list_series(series_id):
        raise HTTPException(404, f"series {series_id!r} not found")
    return _graph_service(services).get(series_id)


@router.post("/series/{series_id}/rebuild", response_model=CommitmentGraph)
def rebuild(series_id: str, services: Services = Depends(get_services)):
    if not services.repo.list_series(series_id):
        raise HTTPException(404, f"series {series_id!r} not found")
    return _graph_service(services).build(series_id)


@router.get("/series/{series_id}/reversals", response_model=list[DecisionReversal])
def get_reversals(series_id: str, services: Services = Depends(get_services)):
    if not services.repo.list_series(series_id):
        raise HTTPException(404, f"series {series_id!r} not found")
    return _graph_service(services).get(series_id).reversals


@router.get("/people/{person_id}/ledger", response_model=PersonLedger)
def get_ledger(person_id: str, services: Services = Depends(get_services)):
    ledger = _graph_service(services).ledger(person_id)
    if not ledger.entries and ledger.display_name is None:
        raise HTTPException(404, f"no commitments recorded for {person_id!r}")
    return ledger
