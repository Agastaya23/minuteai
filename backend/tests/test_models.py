from datetime import date

import pytest
from pydantic import ValidationError

from minuteai.models import (
    ActionItem,
    Decision,
    EvidenceSpan,
    GroundedSentence,
    Meeting,
    MeetingSummary,
    Topic,
    Transcript,
)


def test_transcript_roundtrip(transcript):
    restored = Transcript.model_validate_json(transcript.model_dump_json())
    assert restored == transcript
    assert restored.utterances[7].words[0].text == "Great."


def test_utt_ids_sequential(transcript):
    assert [u.utt_id for u in transcript.utterances] == list(range(20))


def test_word_timestamps_monotonic(transcript):
    last = -1.0
    for u in transcript.utterances:
        for w in u.words:
            assert w.start >= last
            assert w.end > w.start
            last = w.start


def test_window_inclusive(transcript):
    window = transcript.window(3, 5)
    assert [u.utt_id for u in window] == [3, 4, 5]
    assert transcript.window(18, 99)[-1].utt_id == 19


def test_text_between(transcript):
    u = transcript.utterances[6]
    text = transcript.text_between(u.start + 0.01, u.end - 0.01)
    assert "Postgres 16" in text
    assert transcript.text_between(-10, -5) == ""


def test_total_words(transcript):
    assert transcript.total_words() == sum(len(u.text.split()) for u in transcript.utterances)


def test_get(transcript):
    assert transcript.get(4).utt_id == 4
    assert transcript.get(999) is None


def test_evidence_quote_truncated():
    span = EvidenceSpan(utt_id=0, start_sec=0, end_sec=1, speaker_label="S", quote="x" * 500)
    assert len(span.quote) == 200


def test_evidence_rejects_negative_utt():
    with pytest.raises(ValidationError):
        EvidenceSpan(utt_id=-1, start_sec=0, end_sec=1, speaker_label="S", quote="q")


def test_action_item_unresolved_owner_allowed(transcript, span):
    item = ActionItem(
        item_id="a1",
        description="Ask the fraud team for their latency budget",
        owner=None,
        owner_resolution_method="unresolved",
        evidence=[span(transcript, 14)],
    )
    restored = ActionItem.model_validate_json(item.model_dump_json())
    assert restored.owner is None
    assert restored.owner_resolution_method == "unresolved"


def test_summary_roundtrip(transcript, span):
    summary = MeetingSummary(
        meeting_id=transcript.meeting_id,
        abstract=[GroundedSentence(text="Team moves to Postgres.", evidence=[span(transcript, 6)])],
        topics=[Topic(topic_id="t0", title="DB", start_utt=0, end_utt=9)],
        decisions=[
            Decision(
                decision_id="d0",
                statement="Move payments to Postgres 16",
                participants=["SPEAKER_00"],
                evidence=[span(transcript, 6)],
                confidence=0.9,
                status="agreed",
            )
        ],
        action_items=[
            ActionItem(
                item_id="a0",
                description="Write migration script",
                owner="Rahul",
                owner_confidence=0.95,
                owner_resolution_method="self_commitment",
                due_date=date(2026, 10, 30),
                due_raw="by Friday",
                evidence=[span(transcript, 7)],
                confidence=0.9,
            )
        ],
        participants=["Priya", "Rahul", "Ananya"],
        model_version="test",
    )
    assert MeetingSummary.model_validate_json(summary.model_dump_json()) == summary


def test_decision_confidence_bounds():
    with pytest.raises(ValidationError):
        Decision(decision_id="d", statement="s", confidence=1.5)


def test_meeting_defaults():
    m = Meeting(meeting_id="m1", title="x")
    assert m.status == "uploaded"
    assert Meeting.model_validate_json(m.model_dump_json()) == m
