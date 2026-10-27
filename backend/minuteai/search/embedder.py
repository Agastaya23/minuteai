"""Text embeddings.

Default: sentence-transformers `all-MiniLM-L6-v2` (local, free, 384-d).
Fallback when sentence-transformers is not installed: a deterministic hashing
embedder over word unigrams/bigrams and character trigrams. It is weaker but
keeps chunking, dedup, linking and search working (and tests hermetic).
"""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from minuteai.models.transcript import Utterance

log = logging.getLogger(__name__)

WINDOW = 5
STRIDE = 2
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by for from has have i i'm in is it its of on or so that the "
    "this to was we were will with you our us they them he she his her my me do does did "
    "can could should would just okay yes no right sure um uh like".split()
)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    name = "hashing-v1"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOP]
        feats: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(words, words[1:], strict=False)]
        for w in words:
            padded = f"#{w}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.3) for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, h % self.dim] += sign * weight
        return _normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name.split("/")[-1]
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = int(dim_fn())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


@lru_cache
def get_embedder(prefer: str = "auto") -> Embedder:
    if prefer in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except (ImportError, OSError) as e:
            if prefer != "auto":
                raise
            log.info("sentence-transformers unavailable (%s); using hashing embedder", e)
    return HashingEmbedder()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def utterance_windows(
    utterances: list[Utterance], window: int = WINDOW, stride: int = STRIDE
) -> list[list[Utterance]]:
    """Sliding windows of `window` utterances with `stride`; the tail is always covered."""
    if not utterances:
        return []
    if len(utterances) <= window:
        return [utterances]
    starts = list(range(0, len(utterances) - window + 1, stride))
    if starts[-1] + window < len(utterances):
        starts.append(len(utterances) - window)
    return [utterances[s: s + window] for s in starts]
