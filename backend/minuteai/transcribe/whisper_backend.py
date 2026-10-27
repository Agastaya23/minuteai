"""faster-whisper backend (default). Word timestamps are mandatory downstream."""

from __future__ import annotations

import logging
from pathlib import Path

from minuteai.config import get_settings
from minuteai.models.transcript import Transcript, Word
from minuteai.transcribe.base import TranscriptionBackend, TranscriptionError
from minuteai.transcribe.diarization import (
    PyannoteDiarizer,
    SpeakerTurn,
    assign_speakers,
    group_utterances,
)

log = logging.getLogger(__name__)


class WhisperBackend(TranscriptionBackend):
    name = "whisper"

    def __init__(
        self,
        model_size: str | None = None,
        diarizer: PyannoteDiarizer | None = None,
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size or get_settings().whisper_model
        self.diarizer = diarizer if diarizer is not None else PyannoteDiarizer()
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise TranscriptionError(
                    "faster-whisper is not installed: run `make install-ml`"
                ) from e
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe_words(self, audio_path: Path) -> tuple[list[Word], str, float]:
        model = self._load()
        # `segments` is a lazy generator: faster-whisper decodes 30 s windows
        # incrementally, so long recordings are never held fully in memory.
        segments, info = model.transcribe(
            str(audio_path), word_timestamps=True, vad_filter=True, beam_size=5
        )
        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if not text:
                    continue
                start = max(float(w.start), words[-1].start if words else 0.0)
                end = max(float(w.end), start + 0.01)
                words.append(
                    Word(text=text, start=round(start, 3), end=round(end, 3),
                         confidence=round(float(w.probability), 4))
                )
        return words, info.language, float(info.duration)

    def transcribe(self, audio_path: Path, meeting_id: str) -> Transcript:
        words, language, duration = self.transcribe_words(audio_path)
        turns: list[SpeakerTurn] = []
        embeddings: dict[str, list[float]] = {}
        if self.diarizer.available:
            turns = self.diarizer.diarize(audio_path)
            embeddings = self.diarizer.speaker_embeddings(audio_path, turns)
        else:
            log.warning("diarization unavailable (HF_TOKEN or pyannote missing): one speaker")
        utterances = group_utterances(assign_speakers(words, turns))
        return Transcript(
            meeting_id=meeting_id,
            utterances=utterances,
            language=language,
            duration_sec=round(duration, 3),
            backend="whisper",
            model_version=f"faster-whisper/{self.model_size}"
            + ("+pyannote-3.1" if turns else ""),
            speaker_embeddings=embeddings,
        )
