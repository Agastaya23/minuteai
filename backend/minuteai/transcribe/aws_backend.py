"""AWS Transcribe adapter. Normalizes output into the identical Transcript shape."""

from __future__ import annotations

import json
import time
import urllib.request
import uuid
from pathlib import Path

from minuteai.models.transcript import Transcript, Word
from minuteai.storage.aws_clients import transcribe_client
from minuteai.storage.s3_store import S3Store
from minuteai.transcribe.base import TranscriptionBackend, TranscriptionError
from minuteai.transcribe.diarization import SpeakerTurn, assign_speakers, group_utterances


def parse_aws_transcript(payload: dict) -> tuple[list[Word], list[SpeakerTurn]]:
    """Convert an AWS Transcribe result JSON into words + speaker turns."""
    results = payload["results"]
    words: list[Word] = []
    for item in results.get("items", []):
        alt = item["alternatives"][0]
        if item["type"] == "punctuation":
            if words:
                words[-1] = words[-1].model_copy(update={"text": words[-1].text + alt["content"]})
            continue
        words.append(
            Word(
                text=alt["content"],
                start=float(item["start_time"]),
                end=float(item["end_time"]),
                confidence=float(alt.get("confidence", 1.0)),
            )
        )
    turns = [
        SpeakerTurn(
            start=float(seg["start_time"]),
            end=float(seg["end_time"]),
            speaker=_normalize_label(seg["speaker_label"]),
        )
        for seg in results.get("speaker_labels", {}).get("segments", [])
    ]
    return words, turns


def _normalize_label(label: str) -> str:
    # AWS emits "spk_0"; pyannote emits "SPEAKER_00". Normalize to the latter.
    num = label.rsplit("_", 1)[-1]
    return f"SPEAKER_{int(num):02d}" if num.isdigit() else label


class AWSTranscribeBackend(TranscriptionBackend):
    name = "aws_transcribe"

    def __init__(self, max_speakers: int = 10, poll_sec: float = 5.0, timeout_sec: float = 3600):
        self.max_speakers = max_speakers
        self.poll_sec = poll_sec
        self.timeout_sec = timeout_sec
        self.s3 = S3Store()
        self.client = transcribe_client()

    def transcribe(self, audio_path: Path, meeting_id: str) -> Transcript:
        key = f"transcribe-input/{meeting_id}/{audio_path.name}"
        self.s3.upload_file(audio_path, key)
        job = f"minuteai-{meeting_id}-{uuid.uuid4().hex[:8]}"
        self.client.start_transcription_job(
            TranscriptionJobName=job,
            Media={"MediaFileUri": f"s3://{self.s3.bucket}/{key}"},
            MediaFormat=audio_path.suffix.lstrip(".") or "wav",
            IdentifyLanguage=True,
            Settings={"ShowSpeakerLabels": True, "MaxSpeakerLabels": self.max_speakers},
        )
        deadline = time.monotonic() + self.timeout_sec
        while True:
            status = self.client.get_transcription_job(TranscriptionJobName=job)
            tj = status["TranscriptionJob"]
            state = tj["TranscriptionJobStatus"]
            if state == "COMPLETED":
                break
            if state == "FAILED":
                raise TranscriptionError(tj.get("FailureReason", "AWS Transcribe failed"))
            if time.monotonic() > deadline:
                raise TranscriptionError(f"AWS Transcribe job {job} timed out")
            time.sleep(self.poll_sec)
        with urllib.request.urlopen(tj["Transcript"]["TranscriptFileUri"]) as resp:  # noqa: S310
            payload = json.load(resp)
        words, turns = parse_aws_transcript(payload)
        duration = words[-1].end if words else 0.0
        return Transcript(
            meeting_id=meeting_id,
            utterances=group_utterances(assign_speakers(words, turns)),
            language=(tj.get("LanguageCode") or "en").split("-")[0],
            duration_sec=duration,
            backend="aws_transcribe",
            model_version="aws-transcribe",
        )
