import re
from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient

from minuteai.api.app import create_app
from minuteai.api.processing import Services
from minuteai.extract.extractor import Extractor
from minuteai.extract.llm_client import LLMClient
from minuteai.ground.filter import GroundingFilter
from minuteai.models import Meeting
from minuteai.models.search import SearchFilters
from minuteai.samples import SAMPLE_MEETINGS, ReplayProvider, sample_transcript
from minuteai.search.embedder import HashingEmbedder, utterance_windows
from minuteai.search.hybrid import HybridRetriever
from minuteai.search.index import SearchIndex, build_entries
from minuteai.search.service import SearchService
from minuteai.storage import MeetingRepository
from tests.fakes import ScriptedProvider

EMB = HashingEmbedder()
DAYS = {"demo-sprint-planning": 13, "demo-sprint-review": 20, "demo-sprint-retro": 27}


def _seed(repo: MeetingRepository, tmp_path) -> None:
    for mid, spec in SAMPLE_MEETINGS.items():
        t = sample_transcript(mid)
        meeting = Meeting(meeting_id=mid, title=spec["title"], series_id="payments",
                          occurred_at=datetime(2026, 10, DAYS[mid], 10, tzinfo=UTC),
                          status="ready")
        repo.put_meeting(meeting)
        repo.put_transcript(t)
        llm = LLMClient(ReplayProvider(mid), log_path=tmp_path / "c.jsonl")
        summary = Extractor(llm=llm, embedder=EMB).extract_meeting(
            t, occurred_at=meeting.occurred_at)
        kept, _ = GroundingFilter(t, log_path=tmp_path / "d.jsonl").apply(summary)
        repo.put_summary(kept)


def test_windows_cover_everything(transcript):
    wins = utterance_windows(transcript.utterances, 5, 2)
    assert all(len(w) == 5 for w in wins)
    covered = {u.utt_id for w in wins for u in w}
    assert covered == set(range(20))


def test_build_entries(transcript, span):
    meeting = Meeting(meeting_id=transcript.meeting_id, title="t")
    entries = build_entries(meeting, transcript, None)
    assert {e.artifact_type for e in entries} == {"utterances"}
    assert entries[0].start_utt == 0 and entries[0].end_utt == 4
    assert "Priya" in entries[0].people


@pytest.fixture
def index(transcript):
    idx = SearchIndex(EMB)
    for mid in SAMPLE_MEETINGS:
        t = sample_transcript(mid)
        idx.add_meeting(Meeting(meeting_id=mid, title=mid, series_id="payments",
                                occurred_at=datetime(2026, 10, DAYS[mid], tzinfo=UTC)), t, None)
    return idx


def test_hybrid_retrieves_right_meeting(index):
    hits = HybridRetriever(index).retrieve("fraud team latency budget 200 milliseconds", k=3)
    assert hits[0].entry.meeting_id == "demo-sprint-review"
    assert hits[0].bm25_rank is not None and hits[0].dense_rank is not None


def test_filters(index):
    r = HybridRetriever(index)
    only_retro = r.retrieve("staging", SearchFilters(date_from=date(2026, 10, 25)), k=10)
    assert {h.entry.meeting_id for h in only_retro} == {"demo-sprint-retro"}
    assert r.retrieve("staging", SearchFilters(series_id="other")) == []
    by_person = r.retrieve("migration script", SearchFilters(participant="rahul"), k=10)
    assert by_person and all("Rahul" in h.entry.people for h in by_person)


def test_incremental_update_replaces_meeting(index):
    before = len(index)
    t = sample_transcript("demo-sprint-retro")
    index.add_meeting(Meeting(meeting_id="demo-sprint-retro", title="x"), t, None)
    assert len(index) == before
    index.remove_meeting("demo-sprint-retro")
    assert all(e.meeting_id != "demo-sprint-retro" for e in index.entries)


@pytest.mark.usefixtures("aws")
def test_persist_roundtrip(index):
    from minuteai.storage import S3Store

    index.s3 = S3Store()
    index.save()
    loaded = SearchIndex(EMB, s3=S3Store())
    assert loaded.load()
    assert len(loaded) == len(index)
    assert loaded.dense_search("fraud team", 1)[0][0] == index.dense_search("fraud team", 1)[0][0]


def _cite(match: str):
    """LLM fake: cite the excerpt line containing `match`."""

    def respond(prompt: str) -> dict:
        for line in prompt.splitlines():
            m = re.match(r"\[(\d+)\] .*?: (.*)$", line)
            if m and match.lower() in m.group(2).lower():
                return {"answerable": True,
                        "answer": "The fraud team's latency budget is 200 milliseconds.",
                        "evidence": [{"utt_id": int(m.group(1)), "quote": match}]}
        return {"answerable": False, "answer": "Not covered.", "evidence": []}

    return respond


@pytest.fixture
def search_client(tmp_path):
    from moto import mock_aws

    with mock_aws():
        from minuteai.storage.s3_store import S3Store
        from minuteai.storage.schema import create_table

        create_table()
        S3Store().ensure_bucket()
        repo = MeetingRepository()
        _seed(repo, tmp_path)
        provider = ScriptedProvider({"LLMAnswer": _cite("latency budget is 200 milliseconds")})
        search = SearchService(repo, EMB, llm_factory=lambda: LLMClient(
            provider, log_path=tmp_path / "c.jsonl"))
        yield TestClient(create_app(Services(repo=repo, search=search))), provider, tmp_path


def test_search_answer_cites_right_moment(search_client):
    client, _, _ = search_client
    r = client.post("/search", json={"query": "What is the fraud team's latency budget?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answerable"] and body["validated"]
    [ev] = body["evidence"]
    assert ev["meeting_id"] == "demo-sprint-review" and ev["utt_id"] == 4
    assert body["sources"][0]["meeting_title"]


def test_search_unanswerable_says_so(search_client):
    client, provider, _ = search_client
    provider.responses["LLMAnswer"] = {"answerable": False,
                                       "answer": "The meetings do not cover this.",
                                       "evidence": []}
    body = client.post("/search", json={"query": "What is our hiring budget for 2027?"}).json()
    assert body["answerable"] is False and body["evidence"] == []


def test_search_hallucinated_answer_is_replaced(search_client):
    client, provider, tmp_path = search_client
    provider.responses["LLMAnswer"] = {
        "answerable": True, "answer": "The budget is 500 milliseconds.",
        "evidence": [{"utt_id": 999, "quote": "budget is 500 milliseconds"}]}
    body = client.post("/search", json={"query": "fraud latency budget"}).json()
    assert body["answerable"] is False
    assert "UTT_NOT_FOUND" in body["flags"]
    assert "search_answer" in (tmp_path / "data" / "dropped_artifacts.jsonl").read_text()


def test_search_no_results_with_filter(search_client):
    client, _, _ = search_client
    body = client.post("/search", json={"query": "staging",
                                        "filters": {"series_id": "nope"}}).json()
    assert body["answerable"] is False and body["sources"] == []
