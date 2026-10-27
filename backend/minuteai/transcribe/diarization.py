"""Speaker diarization (pyannote 3.1) and word→speaker→utterance merging.

The merge is pure Python and backend-agnostic: both the Whisper and the AWS
adapters funnel their words through `assign_speakers` + `group_utterances`,
which is what guarantees sequential utt_ids and consistent boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from minuteai.models.transcript import Utterance, Word

log = logging.getLogger(__name__)

PAUSE_SPLIT_SEC = 1.5
SENTENCE_PAUSE_SEC = 0.3  # single-speaker fallback only
MAX_UTTERANCE_SEC = 30.0
DEFAULT_SPEAKER = "SPEAKER_00"
_NO_SPACE_BEFORE = {".", ",", "?", "!", ";", ":", "%", "'s", "n't"}


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(words: list[Word], turns: list[SpeakerTurn]) -> list[tuple[Word, str]]:
    """Assign each word to the turn with maximum temporal overlap.

    Words that overlap no turn (diarization gaps) take the nearest turn by
    midpoint distance. With no turns at all every word is DEFAULT_SPEAKER.
    Runs in O(W + T) using a moving pointer since both inputs are time-sorted.
    """
    if not turns:
        return [(w, DEFAULT_SPEAKER) for w in words]
    turns = sorted(turns, key=lambda t: t.start)
    out: list[tuple[Word, str]] = []
    lo = 0
    for w in words:
        while lo < len(turns) and turns[lo].end < w.start:
            lo += 1
        best, best_ov = None, 0.0
        j = lo
        while j < len(turns) and turns[j].start <= w.end:
            ov = _overlap(w.start, w.end, turns[j].start, turns[j].end)
            if ov > best_ov:
                best, best_ov = turns[j], ov
            j += 1
        if best is None:
            mid = (w.start + w.end) / 2
            candidates = turns[max(0, lo - 1): lo + 1] or turns[-1:]
            best = min(
                candidates, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))
            )
        out.append((w, best.speaker))
    return out


def join_words(words: list[Word]) -> str:
    text = ""
    for w in words:
        token = w.text.strip()
        if not token:
            continue
        if text and token not in _NO_SPACE_BEFORE:
            text += " "
        text += token
    return text


def _ends_sentence(word: Word) -> bool:
    return word.text.rstrip("\"')]").endswith((".", "?", "!"))


def group_utterances(
    labeled: list[tuple[Word, str]],
    pause_sec: float = PAUSE_SPLIT_SEC,
    sentence_pause_sec: float | None = None,
    max_utt_sec: float = MAX_UTTERANCE_SEC,
) -> list[Utterance]:
    """Group consecutive same-speaker words; split on speaker change or long pause.

    Also splits at a sentence end when the utterance is already longer than
    `max_utt_sec`, and — when `sentence_pause_sec` is set (no diarization
    available) — at any sentence end followed by at least that much silence,
    so a single-speaker fallback still yields citable, utterance-sized units.
    """
    utterances: list[Utterance] = []
    current: list[Word] = []
    speaker: str | None = None

    def flush() -> None:
        if current:
            utterances.append(
                Utterance(
                    utt_id=len(utterances),
                    speaker_label=speaker or DEFAULT_SPEAKER,
                    text=join_words(current),
                    start=current[0].start,
                    end=current[-1].end,
                    words=list(current),
                )
            )

    for word, spk in labeled:
        if current:
            gap = word.start - current[-1].end
            sentence_break = _ends_sentence(current[-1]) and (
                (sentence_pause_sec is not None and gap >= sentence_pause_sec)
                or current[-1].end - current[0].start >= max_utt_sec
            )
            if spk != speaker or gap > pause_sec or sentence_break:
                flush()
                current = []
        current.append(word)
        speaker = spk
    flush()
    return utterances


class PyannoteDiarizer:
    """Thin wrapper over pyannote.audio 3.1. Imported lazily (heavy dependency)."""

    MODEL = "pyannote/speaker-diarization-3.1"
    EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

    def __init__(self, hf_token: str | None = None) -> None:
        from minuteai.config import get_settings

        self.hf_token = hf_token or get_settings().hf_token
        self._pipeline = None

    @property
    def available(self) -> bool:
        if not self.hf_token:
            return False
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(self.MODEL, use_auth_token=self.hf_token)
            if pipeline is None:
                raise RuntimeError(
                    f"Could not load {self.MODEL}. Accept the user agreements for "
                    "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 on "
                    "HuggingFace with the account that owns HF_TOKEN."
                )
            try:
                import torch

                if torch.backends.mps.is_available():
                    pipeline.to(torch.device("mps"))
                elif torch.cuda.is_available():
                    pipeline.to(torch.device("cuda"))
            except (ImportError, RuntimeError) as e:
                log.debug("diarization stays on CPU: %s", e)
            self._pipeline = pipeline
        return self._pipeline

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        annotation = self._load()(str(audio_path))
        return [
            SpeakerTurn(start=float(seg.start), end=float(seg.end), speaker=str(label))
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]

    def speaker_embeddings(
        self, audio_path: Path, turns: list[SpeakerTurn], max_turns_per_speaker: int = 8
    ) -> dict[str, list[float]]:
        """Mean voice embedding per speaker over their longest turns (best-effort)."""
        try:
            import numpy as np
            from pyannote.audio import Inference, Model
            from pyannote.core import Segment

            model = Model.from_pretrained(self.EMBEDDING_MODEL, use_auth_token=self.hf_token)
            inference = Inference(model, window="whole")
        except (ImportError, RuntimeError, OSError) as e:
            log.warning("speaker embeddings unavailable: %s", e)
            return {}
        by_speaker: dict[str, list[SpeakerTurn]] = {}
        for t in turns:
            if t.end - t.start >= 1.0:
                by_speaker.setdefault(t.speaker, []).append(t)
        out: dict[str, list[float]] = {}
        for spk, spk_turns in by_speaker.items():
            longest = sorted(spk_turns, key=lambda t: t.end - t.start, reverse=True)
            vecs = [
                inference.crop(str(audio_path), Segment(t.start, t.end))
                for t in longest[:max_turns_per_speaker]
            ]
            mean = np.mean(np.vstack(vecs), axis=0)
            out[spk] = (mean / (np.linalg.norm(mean) or 1.0)).tolist()
        return out
