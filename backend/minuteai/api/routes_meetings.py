"""Meeting CRUD, upload, processing and artifact endpoints."""

from __future__ import annotations

import re
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from minuteai.api.deps import get_repo, get_services, meeting_or_404
from minuteai.api.processing import Services, process_meeting
from minuteai.api.schemas import (
    AudioUrl,
    CreateMeetingRequest,
    CreateMeetingResponse,
    MeetingList,
    ProcessRequest,
    ProcessResponse,
)
from minuteai.models.artifacts import MeetingSummary
from minuteai.models.grounding import DroppedArtifact
from minuteai.models.meeting import Meeting, MeetingStatus, utcnow
from minuteai.models.transcript import Transcript
from minuteai.storage.repository import MeetingRepository
from minuteai.storage.s3_store import audio_key

router = APIRouter(prefix="/meetings", tags=["meetings"])

AUDIO_TYPES = re.compile(r"^(audio|video)/[\w.+-]+$")
URL_TTL = 3600


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")[:64]


@router.post("", response_model=CreateMeetingResponse, status_code=status.HTTP_201_CREATED)
def create_meeting(body: CreateMeetingRequest, repo: MeetingRepository = Depends(get_repo)):
    if not AUDIO_TYPES.match(body.content_type):
        raise HTTPException(422, f"unsupported content type {body.content_type!r}")
    meeting_id = f"m-{uuid.uuid4().hex[:12]}"
    key = audio_key(meeting_id, body.filename)
    meeting = Meeting(
        meeting_id=meeting_id,
        title=body.title.strip(),
        series_id=slugify(body.series_name) if body.series_name and slugify(body.series_name)
        else None,
        occurred_at=body.occurred_at or utcnow(),
        audio_s3_key=key,
        participants=[p.strip() for p in body.participants if p.strip()],
        status="uploaded",
    )
    repo.put_meeting(meeting)
    url = repo.s3.presigned_upload_url(key, body.content_type, expires=URL_TTL)
    return CreateMeetingResponse(meeting=meeting, upload_url=url, audio_s3_key=key)


@router.post("/{meeting_id}/process", response_model=ProcessResponse,
             status_code=status.HTTP_202_ACCEPTED)
def start_processing(
    meeting_id: str,
    background: BackgroundTasks,
    body: ProcessRequest | None = None,
    services: Services = Depends(get_services),
):
    body = body or ProcessRequest()
    repo = services.repo
    meeting = meeting_or_404(repo, meeting_id)
    if meeting.status in ("transcribing", "extracting"):
        raise HTTPException(409, f"meeting is already {meeting.status}")
    has_transcript = repo.s3.get_transcript(meeting_id) is not None
    if not has_transcript and not (meeting.audio_s3_key and repo.s3.exists(meeting.audio_s3_key)):
        raise HTTPException(409, "audio has not been uploaded yet")
    # Persist the transition before returning so the first poll already sees progress.
    next_status: MeetingStatus = "extracting" if has_transcript and not body.force \
        else "transcribing"
    repo.update_status(meeting_id, next_status)
    background.add_task(process_meeting, services, meeting_id, body.force, body.backend)
    return ProcessResponse(meeting_id=meeting_id, status=next_status)


@router.get("", response_model=MeetingList)
def list_meetings(
    series_id: str | None = Query(default=None),
    status_filter: MeetingStatus | None = Query(default=None, alias="status"),
    repo: MeetingRepository = Depends(get_repo),
):
    if series_id:
        meetings = repo.list_series(series_id)
        if status_filter:
            meetings = [m for m in meetings if m.status == status_filter]
    elif status_filter:
        meetings = repo.list_by_status(status_filter)
    else:
        meetings = repo.list_meetings()
    return MeetingList(meetings=meetings)


@router.get("/{meeting_id}", response_model=Meeting)
def get_meeting(meeting_id: str, repo: MeetingRepository = Depends(get_repo)):
    return meeting_or_404(repo, meeting_id)


@router.delete("/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_meeting(meeting_id: str, services: Services = Depends(get_services)):
    meeting_or_404(services.repo, meeting_id)
    services.repo.delete_meeting(meeting_id)
    if services.search is not None:
        services.search.remove_meeting(meeting_id)


@router.get("/{meeting_id}/transcript", response_model=Transcript)
def get_transcript(meeting_id: str, repo: MeetingRepository = Depends(get_repo)):
    meeting_or_404(repo, meeting_id)
    transcript = repo.get_transcript(meeting_id)
    if transcript is None:
        raise HTTPException(404, "transcript not available yet")
    return transcript


@router.get("/{meeting_id}/summary", response_model=MeetingSummary)
def get_summary(meeting_id: str, repo: MeetingRepository = Depends(get_repo)):
    meeting_or_404(repo, meeting_id)
    summary = repo.get_summary(meeting_id)
    if summary is None:
        raise HTTPException(404, "summary not available yet")
    return summary


@router.get("/{meeting_id}/dropped", response_model=list[DroppedArtifact])
def get_dropped(meeting_id: str, repo: MeetingRepository = Depends(get_repo)):
    meeting_or_404(repo, meeting_id)
    return repo.get_dropped(meeting_id)


@router.get("/{meeting_id}/audio", response_model=AudioUrl)
def get_audio(meeting_id: str, repo: MeetingRepository = Depends(get_repo)):
    meeting = meeting_or_404(repo, meeting_id)
    if not meeting.audio_s3_key or not repo.s3.exists(meeting.audio_s3_key):
        raise HTTPException(404, "no audio for this meeting")
    return AudioUrl(url=repo.s3.presigned_download_url(meeting.audio_s3_key, URL_TTL),
                    expires_in=URL_TTL)
