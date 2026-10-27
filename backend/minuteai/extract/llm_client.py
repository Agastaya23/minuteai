"""LLMClient — the ONLY place in the codebase that talks to an LLM provider.

Every call:
  * requests structured JSON output from the provider,
  * is parsed into a pydantic model (raw text is never returned),
  * on a parse failure is retried with the validation error appended,
  * backs off exponentially on rate limits / transient server errors,
  * is appended to data/llm_calls.jsonl (cost + latency data for evaluation).

Provider is chosen from LLM_MODEL: ``claude-*`` → Anthropic, ``gemini-*`` → Google.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from minuteai.config import get_settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

SYSTEM_PROMPT = (
    "You extract structured information from meeting transcripts. You only state what the "
    "transcript supports, cite utterance ids for every claim, and return empty lists rather "
    "than inventing content. Respond with JSON matching the requested schema."
)


REQUEST_TIMEOUT_S = 180

MISSING_CREDENTIALS = (
    "No {provider} credentials: set LLM_API_KEY in .env (or ANTHROPIC_API_KEY / GEMINI_API_KEY) "
    "and retry. The transcript is cached, so retrying won't re-transcribe."
)


class LLMError(RuntimeError):
    pass


class LLMParseError(LLMError):
    """The model's output could not be parsed into the response model."""


class LLMRateLimitError(LLMError):
    """Retryable provider error (429, overload, 5xx, connection)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after if retry_after is not None else _parse_retry_after(message)


def _parse_retry_after(message: str) -> float | None:
    """Providers put the server-requested wait in the message ("retry in 31.2s")."""
    m = re.search(r"retry(?:Delay)?['\"]?[:\s]+(?:in\s+)?['\"]?(\d+(?:\.\d+)?)\s*s", message,
                  re.IGNORECASE)
    return float(m.group(1)) if m else None


class RateLimiter:
    """Client-side requests-per-minute cap shared by every client of the same model.

    Free tiers enforce tiny RPM quotas (e.g. 5/min); pacing requests is far cheaper
    than bouncing off 429s. `rpm <= 0` disables it.
    """

    _shared: dict[str, RateLimiter] = {}
    _shared_lock = threading.Lock()

    def __init__(self, rpm: float, clock=time.monotonic, sleep=time.sleep) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    @classmethod
    def for_model(cls, model: str, rpm: float) -> RateLimiter:
        with cls._shared_lock:
            limiter = cls._shared.get(model)
            if limiter is None or limiter.interval != (60.0 / rpm if rpm > 0 else 0.0):
                limiter = cls._shared[model] = cls(rpm)
            return limiter

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            self._sleep(slot - now)


class LLMQuotaExhaustedError(LLMError):
    """A daily/billing quota is used up; retrying soon will not help."""


class LLMRefusalError(LLMError):
    pass


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


class Provider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, response_model: type[BaseModel]) -> ProviderResponse: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 16000) -> None:
        import anthropic

        self._anthropic = anthropic
        # SDK retries disabled: backoff is owned (and logged) by LLMClient.
        try:
            self.client = anthropic.Anthropic(
                api_key=api_key, max_retries=0, timeout=REQUEST_TIMEOUT_S
            ) if api_key else anthropic.Anthropic(max_retries=0, timeout=REQUEST_TIMEOUT_S)
        except anthropic.CredentialsError as e:
            raise LLMError(MISSING_CREDENTIALS.format(provider="Anthropic")) from e
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str, response_model: type[BaseModel]) -> ProviderResponse:
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "format": {"type": "json_schema", "schema": _strict_schema(response_model)}
                },
            )
        except (a.RateLimitError, a.APIConnectionError, a.InternalServerError) as e:
            raise LLMRateLimitError(str(e)) from e
        except a.APIStatusError as e:
            if e.status_code in (408, 409, 429, 529) or e.status_code >= 500:
                raise LLMRateLimitError(str(e)) from e
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except a.CredentialsError as e:
            raise LLMError(MISSING_CREDENTIALS.format(provider="Anthropic")) from e
        except TypeError as e:
            # Older resolution path: TypeError when no credential source resolves.
            if "authentication" in str(e).lower():
                raise LLMError(MISSING_CREDENTIALS.format(provider="Anthropic")) from e
            raise
        if resp.stop_reason == "refusal":
            raise LLMRefusalError(str(getattr(resp, "stop_details", "") or "refused"))
        text = "".join(b.text for b in resp.content if b.type == "text")
        return ProviderResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason,
        )


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, api_key: str | None = None, thinking: str = "low") -> None:
        from google import genai

        self.thinking = thinking

        try:
            # Without a timeout a dropped connection hangs forever; timeouts are retried.
            self.client = genai.Client(api_key=api_key,
                                       http_options={"timeout": REQUEST_TIMEOUT_S * 1000})
        except ValueError as e:  # raised when no API key is found
            raise LLMError(MISSING_CREDENTIALS.format(provider="Gemini")) from e
        self.model = model

    def generate(self, prompt: str, response_model: type[BaseModel]) -> ProviderResponse:
        import httpx
        from google.genai import errors

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_json_schema": response_model.model_json_schema(),
                    **self._thinking_config(),
                },
            )
        except errors.APIError as e:
            if e.code == 429 and "PerDay" in str(e):
                # Daily quotas don't recover by waiting minutes: fail fast and say why.
                limit = re.search(r"quotaValue': '(\d+)'", str(e))
                raise LLMQuotaExhaustedError(
                    f"Gemini daily quota exhausted for {self.model}"
                    + (f" ({limit.group(1)} requests/day on the free tier)" if limit else "")
                    + ". It resets at midnight Pacific time; or set LLM_MODEL to another "
                    "model (quotas are per model)."
                ) from e
            if e.code in (429, 500, 502, 503, 504):
                raise LLMRateLimitError(str(e)) from e
            raise LLMError(f"Gemini API error {e.code}: {e}") from e
        except httpx.TransportError as e:  # DNS/connection blips are transient
            raise LLMRateLimitError(f"connection error: {e}") from e
        usage = resp.usage_metadata
        thoughts = getattr(usage, "thoughts_token_count", 0) or 0
        return ProviderResponse(
            text=resp.text or "",
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            # thinking tokens are billed as output
            output_tokens=(getattr(usage, "candidates_token_count", 0) or 0) + thoughts,
            stop_reason=str(resp.candidates[0].finish_reason) if resp.candidates else None,
        )


    def _thinking_config(self) -> dict:
        if not self.thinking or self.thinking == "default":
            return {}
        if self.model.startswith("gemini-2"):
            budget = {"minimal": 0, "low": 512, "medium": 2048, "high": 8192}
            return {"thinking_config": {"thinking_budget": budget.get(self.thinking, 512)}}
        return {"thinking_config": {"thinking_level": self.thinking.upper()}}


def _strict_schema(model: type[BaseModel]) -> dict:
    """JSON schema with additionalProperties=false on every object (structured outputs)."""
    schema = model.model_json_schema()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node.setdefault("required", list(node.get("properties", {})))
            if "type" in node or "anyOf" in node:
                node.pop("default", None)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


def make_provider(model: str | None = None, api_key: str | None = None) -> Provider:
    s = get_settings()
    model = model or s.llm_model
    api_key = api_key or s.llm_api_key or None
    if model.startswith("gemini"):
        return GeminiProvider(model, api_key, s.llm_thinking)
    if model.startswith("claude"):
        return AnthropicProvider(model, api_key)
    raise ValueError(f"LLM_MODEL {model!r}: expected a claude-* or gemini-* model id")


class LLMClient:
    def __init__(
        self,
        provider: Provider | None = None,
        log_path: Path | None = None,
        max_backoff_attempts: int = 8,
        base_delay: float = 2.0,
        max_delay: float = 120.0,
        sleep=time.sleep,
        rpm: float | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider or make_provider(model)
        self.log_path = log_path or get_settings().data_path("llm_calls.jsonl")
        self.max_backoff_attempts = max_backoff_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self._lock = threading.Lock()
        rpm = get_settings().llm_rpm if rpm is None else rpm
        self.limiter = RateLimiter.for_model(self.provider.model, rpm) if rpm > 0 else None

    @property
    def model(self) -> str:
        return self.provider.model

    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        max_retries: int = 2,
        purpose: str = "",
        meeting_id: str | None = None,
    ) -> T:
        """Call the LLM and parse into `response_model`.

        `max_retries` is the total number of attempts allowed for parse failures:
        the first call plus one retry carrying the validation error (default 2).
        """
        attempt_prompt = prompt
        last_error: Exception | None = None
        for attempt in range(max(1, max_retries)):
            started = time.perf_counter()
            resp: ProviderResponse | None = None
            try:
                resp = self._generate_with_backoff(attempt_prompt, response_model)
                parsed = response_model.model_validate_json(_strip_fences(resp.text))
            except (ValidationError, ValueError) as e:
                last_error = e
                self._log(prompt, resp, started, False, attempt, purpose, meeting_id, repr(e))
                attempt_prompt = (
                    f"{prompt}\n\nYour previous response failed validation:\n{_short(e)}\n"
                    "Return ONLY JSON that matches the schema exactly."
                )
                continue
            except LLMError as e:
                self._log(prompt, resp, started, False, attempt, purpose, meeting_id, repr(e))
                raise
            self._log(prompt, resp, started, True, attempt, purpose, meeting_id, None)
            return parsed
        raise LLMParseError(f"could not parse {response_model.__name__}: {_short(last_error)}")

    def _generate_with_backoff(self, prompt: str, response_model: type[BaseModel]):
        for n in range(self.max_backoff_attempts):
            if self.limiter:
                self.limiter.wait()
            try:
                return self.provider.generate(prompt, response_model)
            except LLMRateLimitError as e:
                if n == self.max_backoff_attempts - 1:
                    raise
                delay = self.base_delay * (2**n) * (1 + random.random() * 0.25)
                if e.retry_after:  # honour the server-requested wait
                    delay = max(delay, e.retry_after + 1.0)
                delay = min(delay, self.max_delay)
                log.warning("LLM rate limited; retrying in %.1fs", delay)
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _log(self, prompt, resp, started, success, retry, purpose, meeting_id, error) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "provider": self.provider.name,
            "model": self.provider.model,
            "purpose": purpose,
            "meeting_id": meeting_id,
            "input_tokens": resp.input_tokens if resp else 0,
            "output_tokens": resp.output_tokens if resp else 0,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "success": success,
            "retry_count": retry,
            "stop_reason": resp.stop_reason if resp else None,
            "error": error,
        }
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text


def _short(e: Exception | None, limit: int = 1500) -> str:
    return str(e)[:limit]
