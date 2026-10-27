"""Transcription backend interface. Backends return our Transcript, nothing else."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from minuteai.models.transcript import Transcript


class TranscriptionError(RuntimeError):
    pass


class TranscriptionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: Path, meeting_id: str) -> Transcript:
        """Transcribe a 16 kHz mono WAV into utterances with word-level timestamps."""


def get_backend(name: str | None = None) -> TranscriptionBackend:
    from minuteai.config import get_settings

    name = name or get_settings().transcribe_backend
    if name == "whisper":
        from minuteai.transcribe.whisper_backend import WhisperBackend

        return WhisperBackend()
    if name == "aws_transcribe":
        from minuteai.transcribe.aws_backend import AWSTranscribeBackend

        return AWSTranscribeBackend()
    raise ValueError(f"unknown transcription backend {name!r}")
