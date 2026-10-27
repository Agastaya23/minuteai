import json
from datetime import date

import pytest

from minuteai.extract.extractor import Extractor
from minuteai.extract.llm_client import LLMClient
from minuteai.ground.filter import GroundingFilter, Strictness, reason_breakdown
from minuteai.ground.validator import GroundingValidator, normalize
from minuteai.models import (
    ActionItem,
    Decision,
    EvidenceSpan,
    GroundedSentence,
    MeetingSummary,
    Topic,
)
from minuteai.samples import ReplayProvider
from minuteai.search.embedder import HashingEmbedder


@pytest.fixture
def v(transcript):
    return GroundingValidator(transcript, use_spacy=False)


def _reasons(v, artifact):
    return v.reasons(artifact)


# -- the passing case -----------------------------------------------------------

def test_fully_grounded_action_item_passes(v, transcript, span):
    item = ActionItem(
        item_id="a0", description="Write the migration script", owner="Rahul",
        owner_speaker_label="SPEAKER_01", owner_resolution_method="self_commitment",
        due_raw="by Friday",
        evidence=[span(transcript, 7, "I will write the migration script by Friday")],
    )
    result = v.validate(item)
    assert result.passed, result.failures
    assert result.confidence == 1.0


def test_paraphrase_tolerated(v, transcript, span):
    s = GroundedSentence(
        text="Postgres has better JSON support.",
        evidence=[span(transcript, 4, "but Postgres gives us a better JSON support")],
    )
    assert v.validate(s).passed


# -- each failure mode, individually --------------------------------------------

def test_utt_not_found(v, transcript, span):
    bad = span(transcript, 6).model_copy(update={"utt_id": 999})
    s = GroundedSentence(text="We will move the payments service.", evidence=[bad])
    assert _reasons(v, s) == {"UTT_NOT_FOUND"}


def test_quote_not_found(v, transcript, span):
    s = GroundedSentence(
        text="The team chose Postgres 16.",
        evidence=[span(transcript, 6, "we unanimously picked Postgres after a long debate")],
    )
    assert _reasons(v, s) == {"QUOTE_NOT_FOUND"}


def test_timestamp_mismatch(v, transcript, span):
    good = span(transcript, 6)
    shifted = good.model_copy(update={"start_sec": good.start_sec + 30,
                                      "end_sec": good.end_sec + 30})
    s = GroundedSentence(text="We will move the payments service to Postgres 16.",
                         evidence=[shifted])
    assert _reasons(v, s) == {"TIMESTAMP_MISMATCH"}


def test_timestamp_tolerance(v, transcript, span):
    good = span(transcript, 6)
    nudged = good.model_copy(update={"start_sec": good.start_sec - 0.4})
    s = GroundedSentence(text="We will move the payments service to Postgres 16.",
                         evidence=[nudged])
    assert v.validate(s).passed


def test_speaker_mismatch_owner(v, transcript, span):
    item = ActionItem(
        item_id="a", description="Make the fraud check asynchronous", owner="Rahul",
        owner_speaker_label="SPEAKER_01", owner_resolution_method="explicit_name",
        evidence=[span(transcript, 12, "make the fraud check asynchronous")],  # SPEAKER_02
    )
    assert _reasons(v, item) == {"SPEAKER_MISMATCH"}


def test_speaker_mismatch_by_name(v, transcript, span):
    # owner given only as a name; resolved to a label via speaker naming
    item = ActionItem(
        item_id="a", description="Make the fraud check asynchronous", owner="Rahul",
        owner_resolution_method="explicit_name",
        evidence=[span(transcript, 12, "make the fraud check asynchronous")],
    )
    assert _reasons(v, item) == {"SPEAKER_MISMATCH"}


def test_speaker_mismatch_decision_participant(v, transcript, span):
    d = Decision(decision_id="d", statement="Move the payments service to Postgres 16",
                 participants=["SPEAKER_00", "SPEAKER_02"], evidence=[span(transcript, 6)])
    assert _reasons(v, d) == {"SPEAKER_MISMATCH"}


def test_speaker_mismatch_span_label(v, transcript, span):
    wrong = span(transcript, 6).model_copy(update={"speaker_label": "SPEAKER_02"})
    s = GroundedSentence(text="We will move the payments service to Postgres 16.",
                         evidence=[wrong])
    assert _reasons(v, s) == {"SPEAKER_MISMATCH"}


def test_entity_not_supported_number(v, transcript, span):
    s = GroundedSentence(text="Staging will run on port 6543.",
                         evidence=[span(transcript, 8, "set up the staging database on port")])
    result = v.validate(s)
    assert result.reasons == {"ENTITY_NOT_SUPPORTED"}
    assert "6543" in result.failures[0].detail


def test_entity_not_supported_name(v, transcript, span):
    s = GroundedSentence(text="The checkout calls Stripe synchronously.",
                         evidence=[span(transcript, 11, "It is synchronous right now")])
    assert _reasons(v, s) == {"ENTITY_NOT_SUPPORTED"}


def test_entity_number_words_match_digits(v, transcript, span):
    s = GroundedSentence(text="The p95 latency is 800 milliseconds.",
                         evidence=[span(transcript, 10, "around 800 milliseconds at p95")])
    assert v.validate(s).passed


def test_entity_speaker_name_is_supported(v, transcript, span):
    s = GroundedSentence(text="Rahul will write the migration script.",
                         evidence=[span(transcript, 7, "I will write the migration script")])
    assert v.validate(s).passed


def test_entity_resolved_relative_date(transcript, span):
    v = GroundingValidator(transcript, reference_date=date(2026, 10, 26), use_spacy=False)
    item = ActionItem(
        item_id="a", description="Write the migration script by October 30", owner="Rahul",
        owner_speaker_label="SPEAKER_01", owner_resolution_method="self_commitment",
        evidence=[span(transcript, 7, "I will write the migration script by Friday")],
    )
    assert v.validate(item).passed
    wrong = item.model_copy(update={"description": "Write the migration script by November 3"})
    assert _reasons(v, wrong) == {"ENTITY_NOT_SUPPORTED"}


def test_no_evidence(v):
    result = v.validate(GroundedSentence(text="Something happened.", evidence=[]))
    assert result.reasons == {"NO_EVIDENCE"}
    assert result.confidence == 0.0


def test_confidence_weighted_by_severity(v, transcript, span):
    entity_fail = GroundedSentence(text="Staging uses port 6543.", evidence=[span(transcript, 8)])
    missing = GroundedSentence(
        text="Staging uses port 5432.",
        evidence=[span(transcript, 8).model_copy(update={"utt_id": 500})],
    )
    assert v.validate(entity_fail).confidence > v.validate(missing).confidence


def test_normalize():
    assert normalize("Hello,   WORLD!  It's two") == "hello world it 2"


def test_cross_meeting_evidence(span):
    from minuteai.samples import sample_transcript

    a, b = sample_transcript("demo-sprint-planning"), sample_transcript("demo-sprint-review")
    v = GroundingValidator({a.meeting_id: a, b.meeting_id: b}, use_spacy=False)
    ev = span(b, 1).model_copy(update={"meeting_id": b.meeting_id})
    s = GroundedSentence(text="The migration script is merged.", evidence=[ev])
    assert v.validate(s).passed
    wrong_meeting = ev.model_copy(update={"meeting_id": a.meeting_id})
    assert not v.validate(s.model_copy(update={"evidence": [wrong_meeting]})).passed


# -- filter -------------------------------------------------------------------------

def _summary(transcript, span) -> MeetingSummary:
    good = GroundedSentence(text="We will move the payments service to Postgres 16.",
                            evidence=[span(transcript, 6)])
    flagged = GroundedSentence(text="Staging uses port 6543.", evidence=[span(transcript, 8)])
    missing = GroundedSentence(
        text="We will hire engineers.",
        evidence=[EvidenceSpan(utt_id=999, start_sec=0, end_sec=1, speaker_label="X", quote="q")],
    )
    return MeetingSummary(
        meeting_id=transcript.meeting_id,
        abstract=[good, flagged, missing],
        topics=[Topic(topic_id="t0", title="x", start_utt=0, end_utt=19,
                      summary=[good, missing])],
        decisions=[Decision(decision_id="d0", statement="Adopt Redis", evidence=[])],
        action_items=[ActionItem(
            item_id="a0", description="Make the fraud check asynchronous", owner="Rahul",
            owner_speaker_label="SPEAKER_01", owner_resolution_method="explicit_name",
            evidence=[span(transcript, 12, "make the fraud check asynchronous")],
        )],
    )


@pytest.mark.parametrize("mode,abstract,decisions,actions,n_dropped", [
    (Strictness.STRICT, 1, 0, 0, 4),
    (Strictness.BALANCED, 2, 0, 1, 2),
    (Strictness.PERMISSIVE, 3, 1, 1, 0),
])
def test_filter_modes(transcript, span, tmp_path, mode, abstract, decisions, actions, n_dropped):
    f = GroundingFilter(transcript, mode=mode, log_path=tmp_path / "dropped.jsonl")
    kept, dropped = f.apply(_summary(transcript, span))
    assert (len(kept.abstract), len(kept.decisions), len(kept.action_items)) == (
        abstract, decisions, actions)
    assert len(dropped) == n_dropped
    assert f.report.total == 5 and f.report.dropped == n_dropped
    assert kept.grounding_mode == mode
    # topic summary sentences share the abstract's verdicts and are not double-logged
    assert [s.text for s in kept.topics[0].summary] == [
        s.text for s in kept.abstract if s.text != "Staging uses port 6543."]


def test_filter_flags_and_caps_confidence(transcript, span, tmp_path):
    kept, _ = GroundingFilter(transcript, mode="BALANCED",
                              log_path=tmp_path / "d.jsonl").apply(_summary(transcript, span))
    good, flagged = kept.abstract
    assert good.validated and not good.flags
    assert not flagged.validated and flagged.flags == ["ENTITY_NOT_SUPPORTED"]
    [item] = kept.action_items
    assert item.flags == ["SPEAKER_MISMATCH"] and item.confidence < 0.5 + 1e-9


def test_dropped_log_written(transcript, span, tmp_path):
    log = tmp_path / "dropped.jsonl"
    _, dropped = GroundingFilter(transcript, mode="STRICT", log_path=log).apply(
        _summary(transcript, span))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == len(dropped) == 4
    row = next(r for r in rows if r["artifact_type"] == "decision")
    assert row["text"] == "Adopt Redis"
    assert row["failures"][0]["reason"] == "NO_EVIDENCE"
    assert {"meeting_id", "evidence", "confidence", "mode", "dropped_at"} <= row.keys()
    assert reason_breakdown(dropped)["UTT_NOT_FOUND"] == 1


def test_sample_hallucinations_are_caught(transcript, tmp_path):
    llm = LLMClient(ReplayProvider("demo-sprint-planning"), log_path=tmp_path / "c.jsonl")
    summary = Extractor(llm=llm, embedder=HashingEmbedder()).extract_meeting(transcript)
    kept, dropped = GroundingFilter(transcript, log_path=tmp_path / "d.jsonl").apply(summary)
    dropped_text = {d.text for d in dropped}
    assert "The team also agreed to hire two more backend engineers." in dropped_text
    assert "Adopt Redis to cache fraud risk scores" in dropped_text
    flagged = {s.text: s.flags for s in kept.abstract if s.flags}
    assert any("30 percent" in t for t in flagged)
    clean = [d for d in kept.decisions if d.validated]
    assert len(clean) == 3
