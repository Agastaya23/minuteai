"""Cohen's kappa between two raters on an overlap subset."""

from __future__ import annotations

from collections import Counter


def cohens_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b):
        raise ValueError("rater label lists must be the same length")
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def interpret(kappa: float) -> str:
    """Landis & Koch (1977) bands."""
    if kappa != kappa:  # NaN
        return "n/a"
    for bound, label in ((0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"),
                         (0.8, "substantial")):
        if kappa <= bound:
            return label
    return "almost perfect"
