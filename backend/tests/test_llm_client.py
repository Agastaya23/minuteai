import json

import pytest
from pydantic import BaseModel

from minuteai.extract.llm_client import (
    LLMClient,
    LLMError,
    LLMParseError,
    LLMRateLimitError,
    _strict_schema,
    make_provider,
)
from minuteai.extract.schemas import LLMActionItemsResponse
from tests.fakes import ScriptedProvider, rate_limited


class Pet(BaseModel):
    name: str
    legs: int


def _client(responses, tmp_path):
    provider = ScriptedProvider(responses)
    sleeps: list[float] = []
    client = LLMClient(provider, log_path=tmp_path / "calls.jsonl", sleep=sleeps.append)
    return client, provider, sleeps


def _log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]


def test_success_and_log(tmp_path):
    client, provider, _ = _client({"Pet": {"name": "Rex", "legs": 4}}, tmp_path)
    pet = client.complete_structured("describe", Pet, purpose="test", meeting_id="m1")
    assert pet == Pet(name="Rex", legs=4)
    [rec] = _log(tmp_path)
    assert rec["success"] is True and rec["retry_count"] == 0
    assert rec["model"] == "fake-model" and rec["purpose"] == "test"
    assert rec["meeting_id"] == "m1"
    assert rec["input_tokens"] > 0 and rec["output_tokens"] > 0
    assert rec["latency_ms"] >= 0 and len(rec["prompt_hash"]) == 16


def test_retry_with_validation_error(tmp_path):
    client, provider, _ = _client(
        {"Pet": ['{"name": "Rex", "legs": "four"}', {"name": "Rex", "legs": 4}]}, tmp_path
    )
    assert client.complete_structured("describe", Pet).legs == 4
    assert len(provider.calls) == 2
    assert "failed validation" in provider.calls[1][1]
    assert "legs" in provider.calls[1][1]
    recs = _log(tmp_path)
    assert [r["success"] for r in recs] == [False, True]
    assert [r["retry_count"] for r in recs] == [0, 1]


def test_parse_error_after_retries(tmp_path):
    client, provider, _ = _client({"Pet": "not json at all"}, tmp_path)
    with pytest.raises(LLMParseError):
        client.complete_structured("describe", Pet)
    assert len(provider.calls) == 2
    assert all(r["success"] is False for r in _log(tmp_path))


def test_code_fences_tolerated(tmp_path):
    client, _, _ = _client({"Pet": '```json\n{"name": "Rex", "legs": 4}\n```'}, tmp_path)
    assert client.complete_structured("x", Pet).name == "Rex"


def test_backoff_on_rate_limit(tmp_path):
    client, provider, sleeps = _client(
        {"Pet": [rate_limited(), rate_limited(), {"name": "Rex", "legs": 4}]}, tmp_path
    )
    assert client.complete_structured("x", Pet).name == "Rex"
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]  # exponential


def test_backoff_gives_up(tmp_path):
    client, _, sleeps = _client({"Pet": [rate_limited()]}, tmp_path)
    client.max_backoff_attempts = 3
    with pytest.raises(LLMRateLimitError):
        client.complete_structured("x", Pet)
    assert len(sleeps) == 2
    assert _log(tmp_path)[-1]["success"] is False


def test_non_retryable_error_raises(tmp_path):
    client, provider, _ = _client({"Pet": LLMError("400 bad request")}, tmp_path)
    with pytest.raises(LLMError):
        client.complete_structured("x", Pet)
    assert len(provider.calls) == 1


def test_strict_schema_closes_objects():
    schema = _strict_schema(LLMActionItemsResponse)
    item = schema["$defs"]["LLMActionItem"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])


def test_provider_selection(monkeypatch):
    assert make_provider("claude-haiku-5-5", "k").name == "anthropic"
    assert make_provider("gemini-2.0-flash", "k").name == "gemini"
    with pytest.raises(ValueError):
        make_provider("gpt-4", "k")


def test_missing_anthropic_credentials_message(monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))  # no `ant` profile either
    from minuteai.extract.llm_client import AnthropicProvider

    with pytest.raises(LLMError, match="LLM_API_KEY"):
        # depending on SDK resolution, construction or the first call fails
        provider = AnthropicProvider("claude-haiku-5-5")
        LLMClient(provider, log_path=tmp_path / "c.jsonl").complete_structured("x", Pet)
