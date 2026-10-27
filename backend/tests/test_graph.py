from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient

from minuteai.api.app import create_app
from minuteai.api.processing import Services
from minuteai.extract.speakers import name_speakers
from minuteai.graph.decisions import detect_reversals
from minuteai.graph.entity_resolution import MeetingBundle, PeopleIndex, resolve_people
from minuteai.graph.lifecycle import LifecycleTracker
from minuteai.graph.linker import link_commitments
from minuteai.graph.service import GraphService
from minuteai.graph.similarity import Similarity, is_completion, is_delay, keyword_overlap
from minuteai.models import ActionItem, Decision, Meeting, MeetingSummary
from minuteai.samples import build_transcript
from minuteai.search.embedder import HashingEmbedder
from minuteai.storage import MeetingRepository
from tests.test_search import _seed

EMB = HashingEmbedder()


@pytest.fixture
def series_graph(tmp_path):
    from moto import mock_aws

    with mock_aws():
        from minuteai.storage.s3_store import S3Store
        from minuteai.storage.schema import create_table

        create_table()
        S3Store().ensure_bucket()
        repo = MeetingRepository()
        _seed(repo, tmp_path)
        for m in repo.list_meetings():  # seeded meetings share series "payments"
            repo.put_transcript(name_speakers(repo.get_transcript(m.meeting_id)))
        service = GraphService(repo, EMB)
        yield repo, service, service.build("payments")


def _by_desc(graph, needle):
    return next(c for c in graph.commitments if needle in c.description)


# -- cue detection ------------------------------------------------------------

def test_cues():
    assert is_completion("I finished the migration script, it is merged.")
    assert is_completion("The rollback plan is written and reviewed.")
    assert not is_completion("Staging is not ready yet.")
    assert not is_completion("It is still in progress, about half done.")
    assert is_delay("Staging slips to Monday.")
    assert keyword_overlap("Write the migration script", "the migration script is merged") == 1.0


# -- identity -----------------------------------------------------------------

def test_people_resolved_across_meetings(series_graph):
    _, _, graph = series_graph
    people = {p.person_id: p for p in graph.people}
    assert set(people) == {"priya", "rahul", "ananya"}
    assert all(p.method == "self_introduction" and p.confidence > 0.8 for p in people.values())
    assert len(people["rahul"].speaker_labels) == 3


def _bundle(mid, lines, day, participants=None, summary=None, embeddings=None):
    t = build_transcript(mid, lines)
    if embeddings:
        t = t.model_copy(update={"speaker_embeddings": embeddings})
    m = Meeting(meeting_id=mid, title=mid, series_id="s", participants=participants or [],
                occurred_at=datetime(2026, 10, day, tzinfo=UTC))
    return MeetingBundle(m, t, summary or MeetingSummary(meeting_id=mid))


def test_unnamed_speaker_stays_unresolved():
    b = _bundle("m1", [("SPEAKER_00", "Let us begin."), ("SPEAKER_01", "Sounds good.")], 1)
    people = resolve_people([b])
    assert all(p.method == "unresolved" and p.confidence == 0 for p in people.values())


def test_voice_embedding_fallback():
    named = _bundle("m1", [("SPEAKER_00", "Hi, I'm Priya."), ("SPEAKER_01", "Hello.")], 1,
                    embeddings={"SPEAKER_00": [1.0, 0.0, 0.0], "SPEAKER_01": [0.0, 1.0, 0.0]})
    anon = _bundle("m2", [("SPEAKER_03", "Let us begin."), ("SPEAKER_04", "Okay.")], 2,
                   embeddings={"SPEAKER_03": [0.97, 0.1, 0.0], "SPEAKER_04": [0.0, 0.0, 1.0]})
    people = resolve_people([named, anon])
    assert people["priya"].speaker_labels == {"m1": "SPEAKER_00", "m2": "SPEAKER_03"}
    assert people["unresolved-m2-speaker_04"].method == "unresolved"


def test_participant_list_corroborates():
    b = _bundle("m1", [("SPEAKER_00", "Thanks Rahul."), ("SPEAKER_01", "Sure.")], 1)
    # "Thanks Rahul" names the *previous* speaker; there is none, so nothing resolves
    assert all(p.method == "unresolved" for p in resolve_people([b]).values())
    b2 = _bundle("m1", [("SPEAKER_01", "Done."), ("SPEAKER_00", "Thanks Rahul.")], 1,
                 participants=["Rahul", "Priya"])
    people = resolve_people([b2])
    assert people["rahul"].method == "participant_list"


# -- lifecycle ------------------------------------------------------------------

def test_fulfilled(series_graph):
    c = _by_desc(series_graph[2], "migration script")
    assert c.state == "FULFILLED" and c.owner_id == "rahul"
    t = c.transitions[-1]
    assert t.meeting_id == "demo-sprint-review" and t.evidence[0].utt_id == 1
    assert t.evidence[0].meeting_id == "demo-sprint-review"


def test_slipped_then_fulfilled(series_graph):
    c = _by_desc(series_graph[2], "staging database")
    assert [t.to_state for t in c.transitions] == ["COMMITTED", "SLIPPED", "FULFILLED"]
    assert c.timeline == {"demo-sprint-planning": "COMMITTED",
                          "demo-sprint-review": "SLIPPED", "demo-sprint-retro": "FULFILLED"}
    assert len(c.mentions) == 2  # linked recurrence across meetings


def test_silently_dropped(series_graph):
    c = _by_desc(series_graph[2], "on-call rotation")
    assert c.state == "SILENTLY_DROPPED"
    assert c.transitions[-1].meeting_id == "demo-sprint-retro"
    assert c.transitions[-1].evidence  # carries the last mention's evidence


def test_unowned_commitment_is_proposed(series_graph):
    c = _by_desc(series_graph[2], "latency budget")
    assert c.owner_id is None
    assert [t.to_state for t in c.transitions] == ["PROPOSED", "FULFILLED"]


def test_different_owner_never_linked(series_graph):
    graph = series_graph[2]
    benchmark = _by_desc(graph, "Benchmark")
    implement = _by_desc(graph, "Implement the asynchronous")
    assert benchmark.commitment_id != implement.commitment_id
    assert implement.owner_id == "ananya" and len(implement.mentions) == 2
    assert implement.state == "SLIPPED"


def test_reversed_by_later_decision():
    sim = Similarity(EMB)
    item = ActionItem(item_id="a0", description="Prepare the Thursday demo", owner="Priya",
                      owner_resolution_method="self_commitment")
    b1 = _bundle("m1", [("SPEAKER_00", "I'm Priya. I will prepare the Thursday demo.")], 1,
                 summary=MeetingSummary(meeting_id="m1", action_items=[item]))
    cancel = Decision(decision_id="d0", statement="Cancel the Thursday demo", status="reversed")
    b2 = _bundle("m2", [("SPEAKER_00", "I'm Priya. We cancel the Thursday demo.")], 8,
                 summary=MeetingSummary(meeting_id="m2", decisions=[cancel]))
    people = PeopleIndex(resolve_people([b1, b2]))
    [c] = link_commitments("s", [b1, b2], people, sim, LifecycleTracker(sim))
    assert c.state == "REVERSED" and c.transitions[-1].meeting_id == "m2"


def test_no_silent_drop_before_due_date():
    sim = Similarity(EMB)
    item = ActionItem(item_id="a0", description="Draft the quarterly report", owner="Priya",
                      owner_resolution_method="self_commitment", due_date=date(2026, 12, 31))
    bundles = [_bundle("m1", [("SPEAKER_00", "I'm Priya.")], 1,
                       summary=MeetingSummary(meeting_id="m1", action_items=[item]))]
    bundles += [_bundle(f"m{i}", [("SPEAKER_00", "Other things.")], i) for i in (2, 3, 4)]
    [c] = link_commitments("s", bundles, PeopleIndex(resolve_people(bundles)), sim)
    assert c.state == "COMMITTED"


# -- decisions ------------------------------------------------------------------

def test_decision_reversal_log(series_graph):
    graph = series_graph[2]
    [r] = graph.reversals
    assert r.original.meeting_id == "demo-sprint-planning"
    assert "Thursday" in r.original.statement and "Cancel" in r.reversing.statement
    assert r.original.evidence and r.reversing.evidence


def test_consistent_decisions_not_reversals():
    sim = Similarity(EMB)
    d1 = Decision(decision_id="d0", statement="Move the payments service to Postgres 16")
    d2 = Decision(decision_id="d0", statement="Keep the payments service on Postgres")
    b1 = _bundle("m1", [("SPEAKER_00", "x")], 1, summary=MeetingSummary(meeting_id="m1",
                                                                        decisions=[d1]))
    b2 = _bundle("m2", [("SPEAKER_00", "x")], 2, summary=MeetingSummary(meeting_id="m2",
                                                                        decisions=[d2]))
    assert detect_reversals("s", [b1, b2], sim) == []


# -- persistence + API ------------------------------------------------------------

def test_graph_persisted(series_graph):
    repo, _, graph = series_graph
    stored = repo.get_graph("payments")
    assert {c.commitment_id for c in stored.commitments} == {
        c.commitment_id for c in graph.commitments}
    staging = next(c for c in stored.commitments if "staging" in c.description)
    assert [t.to_state for t in staging.transitions] == ["COMMITTED", "SLIPPED", "FULFILLED"]


def test_graph_api(series_graph):
    repo, service, _ = series_graph
    client = TestClient(create_app(Services(repo=repo, graph=service)))
    series = client.get("/series").json()
    assert series[0]["series_id"] == "payments" and series[0]["meetings"] == 3
    g = client.get("/series/payments/commitments").json()
    assert len(g["meetings"]) == 3 and g["commitments"]
    assert len(client.get("/series/payments/reversals").json()) == 1
    ledger = client.get("/people/rahul/ledger").json()
    assert ledger["display_name"] == "Rahul"
    assert ledger["counts"]["FULFILLED"] == 2
    assert client.get("/series/nope/commitments").status_code == 404
    assert client.get("/people/nobody/ledger").status_code == 404
    assert client.post("/series/payments/rebuild").status_code == 200
