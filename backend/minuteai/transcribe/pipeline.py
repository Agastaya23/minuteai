"""transcribe_meeting: S3 audio → normalized WAV → backend → persisted Transcript."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from minuteai.models.transcript import Transcript
from minuteai.storage.repository import MeetingRepository
from minuteai.transcribe.audio import to_wav16k
from minuteai.transcribe.base import TranscriptionBackend, get_backend

log = logging.getLogger(__name__)


def transcribe_meeting(
    audio_s3_key: str,
    meeting_id: str,
    backend: TranscriptionBackend | None = None,
    repo: MeetingRepository | None = None,
    force: bool = False,
) -> Transcript:
    repo = repo or MeetingRepository()

    # Transcripts are cached in S3; re-transcribing during development wastes hours.
    if not force:
        cached = repo.s3.get_transcript(meeting_id)
        if cached is not None:
            log.info("using cached transcript for %s", meeting_id)
            if repo.get_transcript(meeting_id) is None:
                repo.put_transcript(cached)
            return cached

    if repo.get_meeting(meeting_id) is not None:
        repo.update_status(meeting_id, "transcribing")
    backend = backend or get_backend()

    with tempfile.TemporaryDirectory(prefix="minuteai-") as tmp:
        tmpdir = Path(tmp)
        raw = repo.s3.get_audio(audio_s3_key, tmpdir / f"input{Path(audio_s3_key).suffix}")
        wav = to_wav16k(raw, tmpdir / "audio16k.wav")
        transcript = backend.transcribe(wav, meeting_id)

    repo.put_transcript(transcript)
    if repo.get_meeting(meeting_id) is not None:
        meeting = repo.require_meeting(meeting_id)
        repo.put_meeting(meeting.model_copy(update={"duration_sec": transcript.duration_sec}))
    return transcript
