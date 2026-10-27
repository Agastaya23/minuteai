"""Token, cost and latency accounting from data/llm_calls.jsonl."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# USD per million tokens (input, output). Update when provider pricing changes.
PRICING = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
}


def load_calls(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def call_cost(rec: dict) -> float:
    inp, out = PRICING.get(rec.get("model", ""), (0.0, 0.0))
    return (rec.get("input_tokens", 0) * inp + rec.get("output_tokens", 0) * out) / 1e6


def summarize(calls: list[dict], meeting_hours: dict[str, float] | None = None) -> dict:
    by_meeting: dict[str, dict] = defaultdict(lambda: {"calls": 0, "input_tokens": 0,
                                                       "output_tokens": 0, "cost_usd": 0.0,
                                                       "latency_ms": 0.0, "failures": 0})
    by_purpose: dict[str, int] = defaultdict(int)
    for c in calls:
        m = by_meeting[c.get("meeting_id") or "—"]
        m["calls"] += 1
        m["input_tokens"] += c.get("input_tokens", 0)
        m["output_tokens"] += c.get("output_tokens", 0)
        m["cost_usd"] += call_cost(c)
        m["latency_ms"] += c.get("latency_ms", 0.0)
        m["failures"] += not c.get("success", True)
        by_purpose[c.get("purpose") or "—"] += 1
    totals = {k: sum(m[k] for m in by_meeting.values())
              for k in ("calls", "input_tokens", "output_tokens", "cost_usd", "latency_ms",
                        "failures")}
    out = {"totals": totals, "by_meeting": dict(by_meeting), "by_purpose": dict(by_purpose),
           "retries": sum(c.get("retry_count", 0) > 0 for c in calls)}
    if meeting_hours:
        hours = sum(meeting_hours.get(mid, 0.0) for mid in by_meeting)
        if hours:
            out["per_meeting_hour"] = {
                "tokens": (totals["input_tokens"] + totals["output_tokens"]) / hours,
                "cost_usd": totals["cost_usd"] / hours,
                "latency_s": totals["latency_ms"] / 1000 / hours,
            }
    return out
