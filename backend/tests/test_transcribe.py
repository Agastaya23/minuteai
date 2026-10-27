import shutil
import subprocess
from pathlib import Path

import pytest

from minuteai.models import Meeting, Transcript, Word
from minuteai.storage import MeetingRepository
from minuteai.transcribe.audio import is_16k_mono_wav, to_wav16k
from minuteai.transcribe.aws_backend import parse_aws_transcript
from minuteai.transcribe.base import TranscriptionBackend
from minuteai.transcribe.diarization import (
    SpeakerTurn,
    assign_speakers,
    group_utterances,
    join_words,
)
from minuteai.transcribe.pipeline import transcribe_meeting


def _w(text, start, end):
    return Word(text=text, start=start, end=end)


def test_assign_speakers_max_overlap():
    words = [_w("a", 0.0, 0.5), _w("b", 0.9, 1.4), _w("c", 2.0, 2.4)]
    turns = [SpeakerTurn(0.0, 1.0, "S0"), SpeakerTurn(1.0, 3.0, "S1")]
    # "b" overlaps S0 by 0.1 and S1 by 0.4 → S1
    assert [s for _, s in assign_speakers(words, turns)] == ["S0", "S1", "S1"]


def test_assign_speakers_gap_takes_nearest():
    words = [_w("x", 5.0, 5.2)]
    turns = [SpeakerTurn(0.0, 1.0, "S0"), SpeakerTurn(5.5, 6.0, "S1")]
    assert assign_speakers(words, turns)[0][1] == "S1"


def test_assign_speakers_no_turns():
    assert assign_speakers([_w("x", 0, 1)], [])[0][1] == "SPEAKER_00"


def test_group_splits_on_speaker_and_pause():
    labeled = [
        (_w("Hello", 0.0, 0.4), "A"), (_w("there.", 0.5, 0.9), "A"),
        (_w("Hi", 1.0, 1.2), "B"),
        (_w("Long", 3.0, 3.3), "B"),  # 1.8s pause → new utterance
        (_w("pause.", 3.4, 3.8), "B"),
    ]
    utts = group_utterances(labeled)
    assert [(u.utt_id, u.speaker_label, u.text) for u in utts] == [
        (0, "A", "Hello there."), (1, "B", "Hi"), (2, "B", "Long pause."),
    ]
    assert utts[2].start == 3.0 and utts[2].end == 3.8


def test_join_words_punctuation():
    assert join_words([_w("Hello", 0, 1), _w(",", 1, 1.1), _w("world", 1, 2)]) == "Hello, world"


AWS_PAYLOAD = {
    "results": {
        "items": [
            {"type": "pronunciation", "start_time": "0.1", "end_time": "0.4",
             "alternatives": [{"content": "Hello", "confidence": "0.99"}]},
            {"type": "punctuation", "alternatives": [{"content": "."}]},
            {"type": "pronunciation", "start_time": "1.0", "end_time": "1.3",
             "alternatives": [{"content": "Hi", "confidence": "0.9"}]},
        ],
        "speaker_labels": {"segments": [
            {"speaker_label": "spk_0", "start_time": "0.0", "end_time": "0.8"},
            {"speaker_label": "spk_1", "start_time": "0.9", "end_time": "1.5"},
        ]},
    }
}


def test_parse_aws_transcript():
    words, turns = parse_aws_transcript(AWS_PAYLOAD)
    assert [w.text for w in words] == ["Hello.", "Hi"]
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"]
    utts = group_utterances(assign_speakers(words, turns))
    assert [u.speaker_label for u in utts] == ["SPEAKER_00", "SPEAKER_01"]


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def _tone(path: Path, rate=44100, channels=2) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"sine=frequency=440:duration=1:sample_rate={rate}", "-ac", str(channels), str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_to_wav16k(tmp_path):
    src = _tone(tmp_path / "in.mp3")
    out = to_wav16k(src, tmp_path / "out.wav")
    assert is_16k_mono_wav(out)
    assert to_wav16k(out, tmp_path / "again.wav") == out  # no-op when already 16k mono


class FakeBackend(TranscriptionBackend):
    name = "fake"

    def __init__(self, transcript: Transcript):
        self.t = transcript
        self.calls = 0

    def transcribe(self, audio_path: Path, meeting_id: str) -> Transcript:
        assert is_16k_mono_wav(audio_path)
        self.calls += 1
        return self.t.model_copy(update={"meeting_id": meeting_id})


@needs_ffmpeg
@pytest.mark.usefixtures("aws")
def test_pipeline_caches(tmp_path, transcript):
    repo = MeetingRepository()
    repo.put_meeting(Meeting(meeting_id="m1", title="t", audio_s3_key="audio/m1/original.mp3"))
    repo.s3.put_audio("audio/m1/original.mp3", _tone(tmp_path / "a.mp3"))
    backend = FakeBackend(transcript)

    t1 = transcribe_meeting("audio/m1/original.mp3", "m1", backend=backend, repo=repo)
    assert t1.meeting_id == "m1"
    assert repo.get_meeting("m1").status == "transcribing"
    assert repo.get_meeting("m1").duration_sec == transcript.duration_sec
    transcribe_meeting("audio/m1/original.mp3", "m1", backend=backend, repo=repo)
    assert backend.calls == 1  # cache hit
    transcribe_meeting("audio/m1/original.mp3", "m1", backend=backend, repo=repo, force=True)
    assert backend.calls == 2
