from datetime import UTC, date, datetime

from minuteai.extract.chunker import (
    chunk_transcript,
    render_utterance,
    segment_topics,
)
from minuteai.extract.dates import resolve_relative_date
from minuteai.extract.extractor import Extractor
from minuteai.extract.llm_client import LLMClient
from minuteai.extract.speakers import detect_names, name_speakers
from minuteai.samples import ReplayProvider, build_transcript, sample_transcript
from minuteai.search.embedder import HashingEmbedder
from tests.fakes import ScriptedProvider

EMB = HashingEmbedder()
MON = date(2026, 10, 26)  # a Monday


def test_render_format(transcript):
    assert render_utterance(transcript.utterances[3]).startswith("[3] SPEAKER_00 (00:")


def test_single_chunk_when_small(transcript):
    chunks = chunk_transcript(transcript, embedder=EMB)
    assert len(chunks) == 1
    assert (chunks[0].start_utt, chunks[0].end_utt) == (0, 19)


def _long_transcript():
    lines = []
    topics = [
        "database migration postgres schema tables index",
        "marketing campaign budget launch social media",
        "hiring interviews candidates recruiter offers",
    ]
    for t, words in enumerate(topics):
        for i in range(30):
            lines.append((f"SPEAKER_0{i % 3}", f"We talked about {words} item {i} for topic {t}."))
    return build_transcript("long", lines)


def test_chunks_preserve_ids_and_overlap():
    t = _long_transcript()
    chunks = chunk_transcript(t, max_tokens=400, overlap_utts=3, embedder=EMB)
    assert len(chunks) > 1
    covered = set()
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start_utt == prev.end_utt - 2  # 3 utterance overlap
    for c in chunks:
        ids = [u.utt_id for u in c.utterances]
        assert ids == list(range(c.start_utt, c.end_utt + 1))  # never renumbered
        covered |= set(ids)
    assert covered == set(range(len(t.utterances)))


def test_topic_segmentation_finds_shifts():
    t = _long_transcript()
    segs = segment_topics(t, embedder=EMB)
    assert segs[0][0] == 0 and segs[-1][1] == 89
    starts = [a for a, _ in segs[1:]]
    assert any(abs(s - 30) <= 3 for s in starts)
    assert any(abs(s - 60) <= 3 for s in starts)


def test_relative_dates():
    assert resolve_relative_date("by Friday", MON) == date(2026, 10, 30)
    assert resolve_relative_date("by Wednesday", MON) == date(2026, 10, 28)
    assert resolve_relative_date("next Monday", MON) == date(2026, 11, 2)
    assert resolve_relative_date("next Friday", date(2026, 10, 27)) == date(2026, 11, 6)
    assert resolve_relative_date("Monday", date(2026, 10, 27)) == date(2026, 11, 2)
    assert resolve_relative_date("tomorrow", MON) == date(2026, 10, 27)
    assert resolve_relative_date("end of the month", MON) == date(2026, 10, 31)
    assert resolve_relative_date("by end of next week", MON) == date(2026, 11, 6)
    assert resolve_relative_date("in two weeks", MON) == date(2026, 11, 9)
    assert resolve_relative_date("November 15", MON) == date(2026, 11, 15)
    assert resolve_relative_date("soon", MON) is None
    assert resolve_relative_date(None, MON) is None


def test_speaker_names(transcript):
    names = detect_names(transcript)
    assert {k: v.name for k, v in names.items()} == {
        "SPEAKER_00": "Priya", "SPEAKER_01": "Rahul", "SPEAKER_02": "Ananya",
    }
    assert names["SPEAKER_00"].method == "self_introduction"
    named = name_speakers(transcript)
    assert named.utterances[7].speaker_name == "Rahul"


def test_speaker_names_restricted_by_participants(transcript):
    assert "SPEAKER_00" not in detect_names(transcript, participants=["Rahul", "Ananya"])


def _replay_extractor(meeting_id, tmp_path):
    llm = LLMClient(ReplayProvider(meeting_id), log_path=tmp_path / "calls.jsonl")
    return Extractor(llm=llm, embedder=EMB)


def test_extract_meeting_attaches_real_evidence(transcript, tmp_path):
    ex = _replay_extractor("demo-sprint-planning", tmp_path)
    summary = ex.extract_meeting(transcript, occurred_at=datetime(2026, 10, 26, tzinfo=UTC))
    assert summary.participants == ["Ananya", "Priya", "Rahul"]
    script = next(a for a in summary.action_items if "migration" in a.description)
    assert script.owner == "Rahul" and script.owner_speaker_label == "SPEAKER_01"
    assert script.due_date == date(2026, 10, 30)
    ev = script.evidence[0]
    u = transcript.utterances[7]
    assert (ev.utt_id, ev.start_sec, ev.end_sec, ev.speaker_label) == (
        7, u.start, u.end, "SPEAKER_01")
    fraud = next(a for a in summary.action_items if "fraud team" in a.description)
    assert fraud.owner is None and fraud.owner_resolution_method == "unresolved"
    assert fraud.owner_confidence == 0.0
    # The hallucinated utt 42 is kept (for the validator to reject), marked UNKNOWN.
    hallucinated = [s for s in summary.abstract if any(e.utt_id == 42 for e in s.evidence)]
    assert hallucinated and hallucinated[0].evidence[0].speaker_label == "UNKNOWN"
    assert [a.item_id for a in summary.action_items] == [f"a{i}" for i in range(5)]
    assert summary.topics and summary.topics[0].start_utt == 0
    assert summary.topics[-1].end_utt == 19


def test_merge_dedups_across_overlapping_chunks(transcript, tmp_path):
    item = {
        "description": "Write the Postgres migration script", "owner": "Rahul",
        "owner_speaker_label": "SPEAKER_01", "owner_resolution_method": "self_commitment",
        "owner_confidence": 0.9, "due_raw": "by Friday", "confidence": 0.9,
        "evidence": [{"utt_id": 7, "quote": "I will write the migration script"}],
    }
    dup = {**item, "description": "Write the migration script for Postgres", "confidence": 0.7}
    provider = ScriptedProvider({
        "LLMSummaryResponse": {"sentences": []},
        "LLMDecisionsResponse": {"decisions": []},
        "LLMActionItemsResponse": {"action_items": [item, dup]},
        "LLMTopicTitlesResponse": {"topics": []},
    })
    ex = Extractor(llm=LLMClient(provider, log_path=tmp_path / "c.jsonl"), embedder=EMB)
    summary = ex.extract_meeting(transcript)
    assert len(summary.action_items) == 1
    assert summary.action_items[0].confidence == 0.9


def test_unresolved_owner_never_guessed(transcript, tmp_path):
    provider = ScriptedProvider({
        "LLMSummaryResponse": {"sentences": []},
        "LLMDecisionsResponse": {"decisions": []},
        "LLMActionItemsResponse": {"action_items": [{
            "description": "Ask the fraud team", "owner": "Rahul",
            "owner_speaker_label": "SPEAKER_01", "owner_resolution_method": "unresolved",
            "owner_confidence": 0.4, "due_raw": None, "confidence": 0.6,
            "evidence": [{"utt_id": 14, "quote": "Someone should ask the fraud team"}],
        }]},
        "LLMTopicTitlesResponse": {"topics": []},
    })
    ex = Extractor(llm=LLMClient(provider, log_path=tmp_path / "c.jsonl"), embedder=EMB)
    [item] = ex.extract_meeting(transcript).action_items
    assert item.owner is None and item.owner_speaker_label is None


def test_baseline_has_no_evidence(transcript, tmp_path):
    provider = ScriptedProvider({"LLMBaselineResponse": {
        "summary": ["They chose Postgres."], "decisions": ["Use Postgres"],
        "action_items": ["Write migration"],
    }})
    ex = Extractor(llm=LLMClient(provider, log_path=tmp_path / "c.jsonl"), embedder=EMB)
    s = ex.extract_meeting_baseline(transcript)
    assert s.abstract[0].evidence == [] and s.action_items[0].evidence == []
    assert len(provider.calls) == 1


def test_all_samples_extract(tmp_path):
    for mid in ("demo-sprint-planning", "demo-sprint-review", "demo-sprint-retro"):
        s = _replay_extractor(mid, tmp_path).extract_meeting(sample_transcript(mid))
        assert s.abstract and s.action_items
