"""Transcript models: the ground truth every artifact is validated against."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    start: float
    end: float
    confidence: float = 1.0


class Utterance(BaseModel):
    utt_id: int = Field(ge=0)
    speaker_label: str
    speaker_name: str | None = None
    text: str
    start: float
    end: float
    words: list[Word] = Field(default_factory=list)

    @property
    def speaker(self) -> str:
        return self.speaker_name or self.speaker_label


class Transcript(BaseModel):
    meeting_id: str
    utterances: list[Utterance]
    language: str = "en"
    duration_sec: float = 0.0
    backend: Literal["whisper", "aws_transcribe", "synthetic"] = "whisper"
    model_version: str = ""

    def get(self, utt_id: int) -> Utterance | None:
        # utt_ids are sequential from 0, so index lookup is the fast path.
        if 0 <= utt_id < len(self.utterances) and self.utterances[utt_id].utt_id == utt_id:
            return self.utterances[utt_id]
        return next((u for u in self.utterances if u.utt_id == utt_id), None)

    def window(self, start_utt: int, end_utt: int) -> list[Utterance]:
        """Utterances with start_utt <= utt_id <= end_utt (inclusive)."""
        return [u for u in self.utterances if start_utt <= u.utt_id <= end_utt]

    def text_between(self, start_sec: float, end_sec: float) -> str:
        """Concatenated text of utterances overlapping [start_sec, end_sec]."""
        return " ".join(
            u.text for u in self.utterances if u.end > start_sec and u.start < end_sec
        )

    def total_words(self) -> int:
        return sum(len(u.words) if u.words else len(u.text.split()) for u in self.utterances)

    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for u in self.utterances:
            seen.setdefault(u.speaker_label, None)
        return list(seen)
