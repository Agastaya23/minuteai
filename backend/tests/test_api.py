import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from minuteai.api.app import create_app
from minuteai.api.processing import Services
from minuteai.extract.extractor import Extractor
from minuteai.extract.llm_client import LLMClient
from minuteai.models import Meeting
from minuteai.samples import ReplayProvider, sample_transcript
from minuteai.search.embedder import HashingEmbedder
from minuteai.storage import MeetingRepository
from tests.test_transcribe import FakeBackend

pytestmark = pytest.mark.usefixtures("aws")


@pytest.fixture
def services(tmp_path):
    repo = MeetingRepository()
    return Services(
        repo=repo,
        extractor=lambda: Extractor(
            llm=LLMClient(ReplayProvider("demo-sprint-planning"),
                          log_path=tmp_path / "calls.jsonl"),
            embedder=HashingEmbedder(),
        ),
        backend=lambda name: FakeBackend(sample_transcript()),
    )


@pytest.fixture
def client(services):
    return TestClient(create_app(services))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["s3"] == "ok" and body["dynamodb"] == "ok"


def test_create_meeting_returns_presigned_url(client, services):
    r = client.post("/meetings", json={
        "title": "Sprint planning", "filename": "rec.m4a", "content_type": "audio/mp4",
        "series_name": "Payments Weekly", "participants": ["Priya", " ", "Rahul"],
    })
    assert r.status_code == 201
    body = r.json()
    m = body["meeting"]
    assert m["status"] == "uploaded" and m["series_id"] == "payments-weekly"
    assert m["participants"] == ["Priya", "Rahul"]
    assert body["audio_s3_key"].endswith("/original.m4a")
    assert "X-Amz-Signature" in body["upload_url"]
    assert services.repo.get_meeting(m["meeting_id"]) is not None


def test_create_meeting_rejects_non_audio(client):
    r = client.post("/meetings", json={"title": "x", "filename": "a.pdf",
                                       "content_type": "application/pdf"})
    assert r.status_code == 422


def test_404s(client):
    for path in ("/meetings/nope", "/meetings/nope/summary", "/meetings/nope/transcript",
                 "/meetings/nope/dropped", "/meetings/nope/audio"):
        assert client.get(path).status_code == 404
    assert client.post("/meetings/nope/process").status_code == 404


def test_process_requires_upload(client):
    mid = client.post("/meetings", json={"title": "x", "filename": "a.wav",
                                         "content_type": "audio/wav"}).json()["meeting"]
    r = client.post(f"/meetings/{mid['meeting_id']}/process")
    assert r.status_code == 409


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_end_to_end_processing(client, services, tmp_path):
    created = client.post("/meetings", json={"title": "Planning", "filename": "a.wav",
                                             "content_type": "audio/wav"}).json()
    mid = created["meeting"]["meeting_id"]
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", str(wav)], check=True)
    services.repo.s3.put_audio(created["audio_s3_key"], wav)

    r = client.post(f"/meetings/{mid}/process", json={})
    assert r.status_code == 202 and r.json()["status"] == "transcribing"
    # TestClient runs background tasks before returning
    meeting = client.get(f"/meetings/{mid}").json()
    assert meeting["status"] == "ready", meeting["error"]
    assert meeting["participants"] == ["Ananya", "Priya", "Rahul"]

    transcript = client.get(f"/meetings/{mid}/transcript").json()
    assert len(transcript["utterances"]) == 20
    summary = client.get(f"/meetings/{mid}/summary").json()
    assert summary["grounding_mode"] == "BALANCED"
    assert all(e["utt_id"] < 20 for s in summary["abstract"] for e in s["evidence"])
    dropped = client.get(f"/meetings/{mid}/dropped").json()
    assert {d["artifact_type"] for d in dropped} == {"abstract", "decision"}
    assert "X-Amz-Signature" in client.get(f"/meetings/{mid}/audio").json()["url"]

    assert [m["meeting_id"] for m in
            client.get("/meetings", params={"status": "ready"}).json()["meetings"]] == [mid]
    assert client.delete(f"/meetings/{mid}").status_code == 204
    assert client.get(f"/meetings/{mid}").status_code == 404


def test_failure_is_persisted(client, services):
    repo = services.repo
    repo.put_meeting(Meeting(meeting_id="m-fail", title="x", audio_s3_key="audio/m-fail/x.wav"))
    repo.s3.put_bytes("audio/m-fail/x.wav", b"not audio")
    assert client.post("/meetings/m-fail/process").status_code == 202
    meeting = client.get("/meetings/m-fail").json()
    assert meeting["status"] == "failed"
    assert meeting["error"]


def test_process_uses_cached_transcript(client, services):
    repo = services.repo
    t = sample_transcript()
    repo.put_meeting(Meeting(meeting_id=t.meeting_id, title="Planning", series_id="s"))
    repo.s3.put_transcript(t)
    r = client.post(f"/meetings/{t.meeting_id}/process")
    assert r.json()["status"] == "extracting"
    assert client.get(f"/meetings/{t.meeting_id}").json()["status"] == "ready"
    assert [m["meeting_id"] for m in
            client.get("/meetings", params={"series_id": "s"}).json()["meetings"]] == [
        t.meeting_id]


def test_cors(client):
    r = client.options("/meetings", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
