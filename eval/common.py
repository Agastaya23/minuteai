"""Shared plumbing for the evaluation harness: paths, caching, meeting records."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
DATA_DIR = EVAL_DIR / "data"
CACHE_DIR = EVAL_DIR / "cache"
RESULTS_DIR = EVAL_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from minuteai.models.transcript import Transcript  # noqa: E402

M = TypeVar("M", bound=BaseModel)


@dataclass
class EvalMeeting:
    """A meeting with whatever gold annotations the dataset provides."""

    meeting_id: str
    transcript: Transcript | None = None  # reference transcript (oracle words)
    audio_path: Path | None = None
    gold_abstract: list[str] = field(default_factory=list)
    gold_actions: list[str] = field(default_factory=list)
    gold_decisions: list[str] = field(default_factory=list)
    gold_action_owners: dict[int, str | None] = field(default_factory=dict)  # by action index
    series_id: str | None = None
    source: str = ""


def cache_key(*parts: object) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cached_model(
    namespace: str, key: str, model: type[M], compute: Callable[[], M], refresh: bool = False
) -> M:
    """Load `model` from eval/cache/<namespace>/<key>.json or compute and store it."""
    path = CACHE_DIR / namespace / f"{key}.json"
    if path.exists() and not refresh:
        return model.model_validate_json(path.read_text())
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json())
    return value


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
