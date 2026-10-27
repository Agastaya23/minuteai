"""Test doubles for the LLM provider."""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel

from minuteai.extract.llm_client import LLMRateLimitError, ProviderResponse


class ScriptedProvider:
    """Returns canned JSON keyed by response-model class name.

    A value may be a dict, a raw string (to simulate malformed output), a
    callable(prompt) -> dict|str, or a list consumed one element per call.
    An Exception instance is raised instead of returned.
    """

    name = "fake"
    model = "fake-model"

    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def generate(self, prompt: str, response_model: type[BaseModel]) -> ProviderResponse:
        self.calls.append((response_model.__name__, prompt))
        value = self.responses[response_model.__name__]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        if isinstance(value, Callable):
            value = value(prompt)
        if isinstance(value, Exception):
            raise value
        text = value if isinstance(value, str) else json.dumps(value)
        return ProviderResponse(text=text, input_tokens=len(prompt) // 4,
                                output_tokens=len(text) // 4, stop_reason="end_turn")


def rate_limited() -> LLMRateLimitError:
    return LLMRateLimitError("429 Too Many Requests")
