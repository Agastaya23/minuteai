from datetime import UTC, datetime, timedelta

import pytest

from minuteai.models import ActionItem, Decision, Meeting, MeetingSummary
from minuteai.models.graph import Commitment, CommitmentGraph, PersonIdentity, Transition
from minuteai.models.grounding import CheckFailure, DroppedArtifact
from minuteai.storage import MeetingRepository, NotFoundError, S3Store
from minuteai.storage.repository import MAX_INLINE_BYTES
from minuteai.storage.schema import create_table, normalize_name

pytestmark = pytest.mark.usefixtures("aws")

T0 = datetime(2026, 10, 1, 10, tzinfo=UTC)


def _meeting(mid: str, days: int = 0, series: str | None = "s1", status="uploaded") -> Meeting:
    return Meeting(
        meeting_id=mid, title=mid, series_id=series, status=status,
        occurred_at=T0 + timedelta(days=days), created_at=T0 + timedelta(days=days),
    )


def test_create_table_idempotent():
    assert create_table() is False


def test_meeting_by_id():  # pattern 1
    repo = MeetingRepository()
    repo.put_meeting(_meeting("m1"))
    assert repo.get_meeting("m1").title == "m1"
    assert repo.get_meeting("nope") is None
    with pytest.raises(NotFoundError):
        repo.require_meeting("nope")


def test_transcript_offloaded_to_s3(transcript):  # pattern 2
    repo = MeetingRepository()
    repo.put_meeting(_meeting(transcript.meeting_id))
    repo.put_transcript(transcript)
    raw = repo._get_raw(f"MEETING#{transcript.meeting_id}", "TRANSCRIPT")
    assert "utterances" not in raw["data"] or '"s3_key"' in raw["data"]
    assert S3Store().exists(f"transcripts/{transcript.meeting_id}.json")
    assert repo.get_transcript(transcript.meeting_id) == transcript
    assert repo.get_transcript("missing") is None


def _summary(transcript, span) -> MeetingSummary:
    return MeetingSummary(
        meeting_id=transcript.meeting_id,
        decisions=[
            Decision(decision_id="d0", statement="Use Postgres", evidence=[span(transcript, 6)])
        ],
        action_items=[
            ActionItem(item_id="a0", description="Migration script", owner="Rahul",
                       owner_resolution_method="self_commitment", evidence=[span(transcript, 7)]),
            ActionItem(item_id="a1", description="Ask fraud team", owner=None,
                       evidence=[span(transcript, 14)]),
        ],
    )


def test_summary_items_and_decisions(transcript, span):  # patterns 3, 4, 5
    repo = MeetingRepository()
    repo.put_meeting(_meeting(transcript.meeting_id))
    summary = _summary(transcript, span)
    repo.put_summary(summary)
    assert repo.get_summary(transcript.meeting_id) == summary
    assert {a.item_id for a in repo.list_action_items(transcript.meeting_id)} == {"a0", "a1"}
    assert [d.decision_id for d in repo.list_decisions(transcript.meeting_id)] == ["d0"]

    # re-putting replaces stale rows rather than accumulating them
    repo.put_summary(summary.model_copy(update={"action_items": summary.action_items[:1]}))
    assert len(repo.list_action_items(transcript.meeting_id)) == 1


def test_large_summary_goes_to_s3(transcript, span):
    repo = MeetingRepository()
    repo.put_meeting(_meeting(transcript.meeting_id))
    summary = _summary(transcript, span)
    summary.participants = ["x" * 1000] * (MAX_INLINE_BYTES // 1000 + 10)
    repo.put_summary(summary)
    assert repo.get_summary(transcript.meeting_id) == summary


def test_list_series_newest_first():  # pattern 6
    repo = MeetingRepository()
    for i in range(3):
        repo.put_meeting(_meeting(f"m{i}", days=i))
    repo.put_meeting(_meeting("other", series="s2"))
    assert [m.meeting_id for m in repo.list_series("s1")] == ["m2", "m1", "m0"]
    assert repo.list_series_ids() == ["s1", "s2"]


def test_series_change_removes_link():
    repo = MeetingRepository()
    repo.put_meeting(_meeting("m0"))
    repo.put_meeting(_meeting("m0", series=None))
    assert repo.list_series("s1") == []


def test_list_by_owner_across_meetings(transcript, span):  # pattern 7
    repo = MeetingRepository()
    for i, mid in enumerate(["ma", "mb"]):
        repo.put_meeting(_meeting(mid, days=i))
        s = _summary(transcript, span).model_copy(update={"meeting_id": mid})
        repo.put_summary(s)
    rows = repo.list_by_owner("rahul")
    assert [mid for mid, _ in rows] == ["mb", "ma"]
    assert all(item.owner == "Rahul" for _, item in rows)
    assert repo.list_by_owner(" RAHUL ") == rows
    assert normalize_name("Dr. A. Kumar") == "dr_a_kumar"


def test_list_by_status_and_update():  # pattern 8
    repo = MeetingRepository()
    repo.put_meeting(_meeting("m1"))
    repo.put_meeting(_meeting("m2", days=1))
    repo.update_status("m1", "transcribing")
    assert [m.meeting_id for m in repo.list_by_status("uploaded")] == ["m2"]
    assert [m.meeting_id for m in repo.list_by_status("transcribing")] == ["m1"]
    failed = repo.update_status("m2", "failed", error="boom")
    assert failed.error == "boom"
    assert repo.get_meeting("m2").status == "failed"
    assert {m.meeting_id for m in repo.list_meetings()} == {"m1", "m2"}


def test_dropped_roundtrip(span, transcript):
    repo = MeetingRepository()
    d = DroppedArtifact(
        meeting_id="m1", artifact_type="decision", artifact_id="d9", text="x",
        evidence=[span(transcript, 0)],
        failures=[CheckFailure(check="existence", reason="UTT_NOT_FOUND", detail="999")],
    )
    repo.put_dropped("m1", [d])
    assert repo.get_dropped("m1") == [d]
    assert repo.get_dropped("none") == []


def test_graph_roundtrip():
    repo = MeetingRepository()
    t = Transition(commitment_id="c1", from_state=None, to_state="COMMITTED",
                   meeting_id="m1", occurred_at=T0, reason="created")
    c = Commitment(commitment_id="c1", series_id="s1", description="d", owner_id="p1",
                   origin_meeting_id="m1", created_at=T0, transitions=[t])
    p = PersonIdentity(person_id="p1", display_name="Rahul", method="self_introduction",
                       confidence=0.9)
    repo.put_graph(CommitmentGraph(series_id="s1", commitments=[c], people=[p]))
    g = repo.get_graph("s1")
    assert g.commitments == [c]
    assert g.people == [p]
    assert repo.list_person_series("p1")[0][0] == "s1"
    assert repo.get_graph("nope") is None


def test_presigned_urls():
    s3 = S3Store()
    assert "X-Amz-Signature" in s3.presigned_upload_url("audio/x.wav", "audio/wav")
    assert "X-Amz-Signature" in s3.presigned_download_url("audio/x.wav")
