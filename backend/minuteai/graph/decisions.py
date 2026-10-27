"""Decision-reversal detection across a series.

A decision in meeting k is reversed by one in meeting k+n when they concern the
same thing (similarity ≥ threshold) and the later one carries an explicit
reversal signal the earlier one lacks: status "reversed", or cancel / scrap /
no longer / instead … in its statement. Both sides' evidence is kept.
"""

from __future__ import annotations

from minuteai.graph.entity_resolution import MeetingBundle
from minuteai.graph.similarity import Similarity, is_reversal
from minuteai.models.artifacts import Decision
from minuteai.models.graph import DecisionRef, DecisionReversal

REVERSAL_SIMILARITY = 0.55


def _ref(b: MeetingBundle, d: Decision) -> DecisionRef:
    return DecisionRef(
        meeting_id=b.meeting_id, decision_id=d.decision_id, statement=d.statement,
        occurred_at=b.meeting.occurred_at,
        evidence=[e.model_copy(update={"meeting_id": b.meeting_id}) for e in d.evidence],
    )


def _signals_reversal(d: Decision) -> bool:
    return d.status == "reversed" or is_reversal(d.statement)


def detect_reversals(
    series_id: str, bundles: list[MeetingBundle], sim: Similarity,
    threshold: float = REVERSAL_SIMILARITY,
) -> list[DecisionReversal]:
    ordered = [b for b in sorted(bundles, key=lambda b: b.meeting.occurred_at) if b.summary]
    reversals: list[DecisionReversal] = []
    reversed_ids: set[tuple[str, str]] = set()
    for j, later in enumerate(ordered):
        for dl in later.summary.decisions:
            if not _signals_reversal(dl):
                continue
            best = None
            for earlier in reversed(ordered[:j]):  # most recent earlier decision first
                for de in earlier.summary.decisions:
                    key = (earlier.meeting_id, de.decision_id)
                    if key in reversed_ids or _signals_reversal(de):
                        continue
                    score = sim.score(de.statement, dl.statement)
                    if score >= threshold and (best is None or score > best[2]):
                        best = (earlier, de, score)
            if best is None:
                continue
            earlier, de, score = best
            reversed_ids.add((earlier.meeting_id, de.decision_id))
            reversals.append(DecisionReversal(
                reversal_id=f"{earlier.meeting_id}-{de.decision_id}--{later.meeting_id}-"
                            f"{dl.decision_id}",
                series_id=series_id,
                original=_ref(earlier, de),
                reversing=_ref(later, dl),
                similarity=round(score, 3),
                reason=f"“{dl.statement}” ({later.meeting.title}) overturns "
                       f"“{de.statement}” ({earlier.meeting.title})",
            ))
    return reversals
