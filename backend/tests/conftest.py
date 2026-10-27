from __future__ import annotations

import os

import pytest

# Make sure no test ever reaches real AWS or a real LLM.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_REGION"] = "us-east-1"
os.environ["DDB_ENDPOINT_URL"] = ""
os.environ["S3_ENDPOINT_URL"] = ""
os.environ["GROUNDING_MODE"] = "BALANCED"
os.environ["TRANSCRIBE_BACKEND"] = "whisper"
os.environ["S3_BUCKET"] = "minuteai-test"
os.environ["DDB_TABLE"] = "minuteai-test"
os.environ["LLM_API_KEY"] = "test-key"

from minuteai.config import get_settings  # noqa: E402
from minuteai.models.evidence import EvidenceSpan  # noqa: E402
from minuteai.models.transcript import Transcript  # noqa: E402
from minuteai.samples import sample_transcript  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def transcript() -> Transcript:
    """Synthetic 20-utterance, 3-speaker sprint planning meeting."""
    t = sample_transcript("demo-sprint-planning")
    assert len(t.utterances) == 20
    return t


@pytest.fixture
def span():
    def make(t: Transcript, utt_id: int, quote: str | None = None) -> EvidenceSpan:
        u = t.utterances[utt_id]
        return EvidenceSpan(
            utt_id=u.utt_id,
            start_sec=u.start,
            end_sec=u.end,
            speaker_label=u.speaker_label,
            quote=quote if quote is not None else u.text,
        )

    return make


@pytest.fixture
def aws(monkeypatch):
    """moto-mocked S3 + DynamoDB with the table and bucket created."""
    from moto import mock_aws

    with mock_aws():
        from minuteai.storage.s3_store import S3Store
        from minuteai.storage.schema import create_table

        create_table()
        S3Store().ensure_bucket()
        yield
