"""End-to-end processing of one meeting. Runs as a FastAPI background task.

uploaded → transcribing → extracting → ready │ failed (error persisted)
After `ready`: the search index is updated and, for series meetings, the
accountability graph is rebuilt. Those are best-effort and never fail the meeting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from minuteai.extract.extractor import Extractor
from minuteai.extract.speakers import name_speakers
from minuteai.ground.filter import GroundingFilter
from minuteai.models.artifacts import MeetingSummary
from minuteai.storage.repository import MeetingRepository
from minuteai.transcribe.base import TranscriptionBackend, get_backend
from minuteai.transcribe.pipeline import transcribe_meeting

if TYPE_CHECKING:
    from minuteai.search.service import SearchService

log = logging.getLogger(__name__)


@dataclass
class Services:
    """Factories for the heavy collaborators, swappable in tests."""

    repo: MeetingRepository
    extractor: Callable[[], Extractor] = Extractor
    backend: Callable[[str | None], TranscriptionBackend] = get_backend
    on_ready: list[Callable[[str], None]] = field(default_factory=list)
    search: SearchService | None = None


class ProcessingError(RuntimeError):
    pass


def process_meeting(
    services: Services,
    meeting_id: str,
    force: bool = False,
    backend_name: str | None = None,
) -> MeetingSummary | None:
    repo = services.repo
    meeting = repo.require_meeting(meeting_id)
    try:
        transcript = None if force else repo.s3.get_transcript(meeting_id)
        if transcript is not None:
            if repo.get_transcript(meeting_id) is None:
                repo.put_transcript(transcript)
        else:
            if not meeting.audio_s3_key:
                raise ProcessingError("meeting has no audio and no cached transcript")
            repo.update_status(meeting_id, "transcribing")
            transcript = transcribe_meeting(
                meeting.audio_s3_key, meeting_id, backend=services.backend(backend_name),
                repo=repo, force=True,
            )

        # Persist speaker names resolved from the dialogue itself (self-introductions,
        # direct address) so the UI, search and graph all see the same names.
        named = name_speakers(transcript, meeting.participants or None)
        if named != transcript:
            transcript = named
            repo.put_transcript(transcript)
        repo.update_status(meeting_id, "extracting", duration_sec=transcript.duration_sec)
        summary = services.extractor().extract_meeting(
            transcript, occurred_at=meeting.occurred_at, participants=meeting.participants or None
        )
        kept, dropped = GroundingFilter(
            transcript, reference_date=meeting.occurred_at.date()
        ).apply(summary)
        repo.put_summary(kept)
        repo.put_dropped(meeting_id, dropped)
        participants = meeting.participants or kept.participants
        repo.update_status(meeting_id, "ready", participants=participants)
    except Exception as e:  # noqa: BLE001 — any failure must be persisted for the UI
        log.exception("processing %s failed", meeting_id)
        repo.update_status(meeting_id, "failed", error=f"{type(e).__name__}: {e}"[:1000])
        return None

    for hook in services.on_ready:
        try:
            hook(meeting_id)
        except Exception:  # noqa: BLE001 — post-processing is best-effort
            log.exception("post-processing hook %s failed for %s", hook, meeting_id)
    return kept
