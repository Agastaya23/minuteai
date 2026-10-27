"""faster-whisper backend (default). Word timestamps are mandatory downstream."""

from __future__ import annotations

import logging
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from minuteai.config import get_settings
from minuteai.models.transcript import Transcript, Word
from minuteai.transcribe.base import TranscriptionBackend, TranscriptionError
from minuteai.transcribe.diarization import (
    SENTENCE_PAUSE_SEC,
    PyannoteDiarizer,
    SpeakerTurn,
    assign_speakers,
    group_utterances,
)

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_CUT_SEARCH_SEC = 15.0
_FRAME_SEC = 0.5


def iter_wav_windows(path: Path, window_sec: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (offset_sec, float32 samples) windows from a 16 kHz mono 16-bit WAV."""
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise TranscriptionError("expected 16 kHz mono 16-bit WAV (run audio.to_wav16k)")
        total = wf.getnframes()
        pos = 0
        carry = np.zeros(0, dtype=np.float32)
        while pos < total or carry.size:
            want = int(window_sec * SAMPLE_RATE) - carry.size
            raw = wf.readframes(max(0, min(want, total - pos)))
            pos += len(raw) // 2
            chunk = np.concatenate(
                [carry, np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0]
            )
            offset = (pos - chunk.size) / SAMPLE_RATE
            if pos >= total:
                if chunk.size:
                    yield offset, chunk
                return
            cut = _quiet_cut(chunk)
            yield offset, chunk[:cut]
            carry = chunk[cut:]


def _quiet_cut(chunk: np.ndarray) -> int:
    """Index of the quietest half-second frame within the last few seconds."""
    frame = int(_FRAME_SEC * SAMPLE_RATE)
    start = max(0, chunk.size - int(_CUT_SEARCH_SEC * SAMPLE_RATE))
    tail = chunk[start:]
    n = tail.size // frame
    if n == 0:
        return chunk.size
    energy = (tail[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    return start + int(np.argmin(energy)) * frame + frame // 2


class WhisperBackend(TranscriptionBackend):
    name = "whisper"

    def __init__(
        self,
        model_size: str | None = None,
        diarizer: PyannoteDiarizer | None = None,
        device: str = "auto",
        compute_type: str = "int8",
        window_sec: float = 600.0,
    ) -> None:
        self.model_size = model_size or get_settings().whisper_model
        self.diarizer = diarizer if diarizer is not None else PyannoteDiarizer()
        self.device = device
        self.compute_type = compute_type
        self.window_sec = window_sec
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
        """Decode the 16 kHz WAV window by window and transcribe each window.

        Windows are ~10 minutes, cut at the quietest half-second near the
        boundary so words are not split; only one window is in memory at a time.
        """
        model = self._load()
        words: list[Word] = []
        languages: dict[str, float] = {}
        duration = 0.0
        for offset, samples in iter_wav_windows(audio_path, self.window_sec):
            duration = offset + len(samples) / SAMPLE_RATE
            segments, info = model.transcribe(
                samples, word_timestamps=True, vad_filter=True, beam_size=5
            )
            languages[info.language] = languages.get(info.language, 0) + len(samples)
            for seg in segments:
                for w in seg.words or []:
                    text = w.word.strip()
                    if not text:
                        continue
                    start = max(float(w.start) + offset, words[-1].start if words else 0.0)
                    end = max(float(w.end) + offset, start + 0.01)
                    words.append(
                        Word(text=text, start=round(start, 3), end=round(end, 3),
                             confidence=round(float(w.probability), 4))
                    )
        language = max(languages, key=languages.get) if languages else "en"
        return words, language, duration

    def transcribe(self, audio_path: Path, meeting_id: str) -> Transcript:
        words, language, duration = self.transcribe_words(audio_path)
        turns: list[SpeakerTurn] = []
        embeddings: dict[str, list[float]] = {}
        if self.diarizer.available:
            turns = self.diarizer.diarize(audio_path)
            embeddings = self.diarizer.speaker_embeddings(audio_path, turns)
        else:
            log.warning("diarization unavailable (HF_TOKEN or pyannote missing): one speaker")
        utterances = group_utterances(
            assign_speakers(words, turns),
            sentence_pause_sec=None if turns else SENTENCE_PAUSE_SEC,
        )
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
