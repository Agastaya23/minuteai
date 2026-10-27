"""Dependency wiring. Everything hangs off app.state so tests can swap it."""

from __future__ import annotations

from fastapi import HTTPException, Request

from minuteai.api.processing import Services
from minuteai.models.meeting import Meeting
from minuteai.storage.repository import MeetingRepository


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_repo(request: Request) -> MeetingRepository:
    return request.app.state.services.repo


def meeting_or_404(repo: MeetingRepository, meeting_id: str) -> Meeting:
    meeting = repo.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail=f"meeting {meeting_id!r} not found")
    return meeting
