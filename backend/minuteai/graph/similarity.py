"""Text similarity and cue detection shared by the linker, lifecycle and reversal logic."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz

from minuteai.search.embedder import _STOP, Embedder

# Verbs that describe *doing* a task rather than *what* the task is; ignored when
# comparing commitments ("write the migration script" vs "finish the migration script").
GENERIC = frozenset(
    "write finish finished set setting up make making do doing get getting start send "
    "own work implement implementing handle take care need needs update updating going "
    "will ready done complete completed task item".split()
)
COMPLETION = re.compile(
    r"\b(finished|done|completed?|merged|shipped|deployed|sent|delivered|resolved|fixed|"
    r"(?:is|are|was|been|got) (?:written|reviewed|live|ready|merged|updated)|went live|"
    r"talked to|spoke (?:to|with)|closed)\b",
    re.IGNORECASE,
)
NOT_DONE = re.compile(r"\b(not (?:yet|ready|done|finished)|still|in progress|half done|"
                      r"haven't|hasn't|isn't|aren't|won't)\b", re.IGNORECASE)
DELAY = re.compile(
    r"\b(slip(?:s|ped)?|delay(?:ed)?|push(?:ed)? (?:back|out)|postpone[d]?|need (?:until|more time)"
    r"|not ready|behind|later than|moved to|expect to finish|by next|until next)\b",
    re.IGNORECASE,
)
REVERSAL = re.compile(
    r"\b(cancel(?:led|ed)?|scrap(?:ped)?|drop(?:ped)?|revert(?:ed)?|revers(?:e|ed)|"
    r"no longer|stop(?:ped)?|abandon(?:ed)?|instead|rather than|undo)\b",
    re.IGNORECASE,
)


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for w in words:
        if w in _STOP or w in GENERIC or len(w) < 2:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)  # crude singular
    return out


def keyword_overlap(a: str, b: str) -> float:
    """Overlap coefficient of content words: |A∩B| / min(|A|, |B|)."""
    ca, cb = content_words(a), content_words(b)
    if not ca or not cb:
        return 0.0
    return len(ca & cb) / min(len(ca), len(cb))


class Similarity:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._cache: dict[str, np.ndarray] = {}

    def vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.embedder.embed([text])[0]
        return self._cache[text]

    def semantic(self, a: str, b: str) -> float:
        return float(np.dot(self.vec(a), self.vec(b)))

    def score(self, a: str, b: str) -> float:
        """Blend of embedding cosine, content-word overlap and fuzzy token match."""
        fz = fuzz.token_set_ratio(" ".join(sorted(content_words(a))),
                                  " ".join(sorted(content_words(b)))) / 100
        return max(self.semantic(a, b), 0.5 * keyword_overlap(a, b) + 0.5 * fz)


@lru_cache(maxsize=4096)
def is_completion(text: str) -> bool:
    return bool(COMPLETION.search(text)) and not NOT_DONE.search(text)


def is_delay(text: str) -> bool:
    return bool(DELAY.search(text))


def is_reversal(text: str) -> bool:
    return bool(REVERSAL.search(text))
