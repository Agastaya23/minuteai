"""Link action items across a series into commitments.

For each action item (meetings in chronological order) decide whether it is a
new commitment or a recurrence of an open one. The match score combines:
  * semantic similarity of descriptions   (embedding cosine / content overlap)
  * owner identity match                  (same resolved person)
  * entity overlap                        (numbers, names, technical terms)
  * temporal ordering                     (only earlier, still-open commitments)
Matching is greedy against open commitments, newest first.
"""

from __future__ import annotations

import re

from minuteai.graph.entity_resolution import MeetingBundle, PeopleIndex
from minuteai.graph.lifecycle import PRIORITY, LifecycleTracker
from minuteai.graph.similarity import Similarity
from minuteai.models.artifacts import ActionItem
from minuteai.models.graph import TERMINAL_STATES, Commitment, CommitmentMention

# With all-MiniLM-L6-v2, ~0.78 cosine is the recommended semantic threshold. The
# blended score below is calibrated so 0.6 corresponds to it for both embedders.
LINK_THRESHOLD = 0.6
W_SEMANTIC, W_OWNER, W_ENTITY = 0.7, 0.2, 0.1


def _entities(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"\b(?:\d[\w.]*|[A-Z][a-zA-Z]+\w*)\b", text)} - {
        "the", "a", "i", "we"}


def owner_of(item: ActionItem, meeting_id: str, people: PeopleIndex) -> str | None:
    return people.for_label(meeting_id, item.owner_speaker_label) or people.for_name(item.owner)


def match_score(
    sim: Similarity, commitment: Commitment, item: ActionItem, owner_id: str | None
) -> float:
    semantic = max(sim.score(m.description, item.description) for m in commitment.mentions)
    if commitment.owner_id and owner_id:
        owner = 1.0 if commitment.owner_id == owner_id else 0.0
    else:
        owner = 0.5  # unknown on either side: neither evidence for nor against
    ea, eb = _entities(commitment.description), _entities(item.description)
    entity = len(ea & eb) / len(ea | eb) if ea | eb else 0.5
    if commitment.owner_id and owner_id and commitment.owner_id != owner_id:
        return 0.0  # different named owners are never the same commitment
    return W_SEMANTIC * semantic + W_OWNER * owner + W_ENTITY * entity


def mention_of(item: ActionItem, bundle: MeetingBundle, owner_name: str | None,
               similarity: float = 1.0) -> CommitmentMention:
    return CommitmentMention(
        meeting_id=bundle.meeting_id, item_id=item.item_id,
        occurred_at=bundle.meeting.occurred_at, description=item.description,
        owner=owner_name or item.owner, due_date=item.due_date, due_raw=item.due_raw,
        evidence=[e.model_copy(update={"meeting_id": bundle.meeting_id}) for e in item.evidence],
        similarity=round(similarity, 3),
    )


def best_match(
    sim: Similarity, open_: list[Commitment], item: ActionItem, owner_id: str | None,
    threshold: float = LINK_THRESHOLD,
) -> tuple[Commitment | None, float]:
    """Greedy: scan open commitments newest first, keep the highest score ≥ threshold."""
    best, best_score = None, 0.0
    for c in sorted(open_, key=lambda c: c.created_at, reverse=True):
        s = match_score(sim, c, item, owner_id)
        if s >= threshold and s > best_score + 1e-9:
            best, best_score = c, s
    return best, best_score


def link_commitments(
    series_id: str,
    bundles: list[MeetingBundle],
    people: PeopleIndex,
    sim: Similarity,
    tracker: LifecycleTracker | None = None,
    threshold: float = LINK_THRESHOLD,
) -> list[Commitment]:
    """Link every action item in the series into commitments and infer their lifecycle.

    Meetings are processed chronologically. For meeting k: open commitments are
    observed against k's dialogue and decisions, k's action items are linked to
    open commitments (or start new ones), then silence/timeline are updated.
    """
    tracker = tracker or LifecycleTracker(sim)
    commitments: list[Commitment] = []
    for bundle in sorted(bundles, key=lambda b: b.meeting.occurred_at):
        existing = list(commitments)
        open_before = [c for c in existing if c.state not in TERMINAL_STATES]
        observations = {c.commitment_id: tracker.observe(c, bundle) for c in open_before}
        created: list[Commitment] = []
        claimed: set[str] = set()

        for item in bundle.summary.action_items if bundle.summary else []:
            owner_id = owner_of(item, bundle.meeting_id, people)
            owner_name = people.name(owner_id) or item.owner
            candidates = [c for c in open_before if c.commitment_id not in claimed]
            match, score = best_match(sim, candidates, item, owner_id, threshold)
            if match is not None:
                match.mentions.append(mention_of(item, bundle, owner_name, score))
                if match.owner_id is None and owner_id:
                    match.owner_id, match.owner_name = owner_id, owner_name
                claimed.add(match.commitment_id)
                rec = tracker.on_recurrence(match, bundle)
                obs = observations[match.commitment_id]
                if PRIORITY[rec.to_state] > PRIORITY[obs.to_state]:
                    obs = rec
                obs.mentioned = True
                observations[match.commitment_id] = obs
                continue
            c = Commitment(
                commitment_id=f"{bundle.meeting_id}-{item.item_id}",
                series_id=series_id,
                description=item.description,
                owner_id=owner_id,
                owner_name=owner_name,
                origin_meeting_id=bundle.meeting_id,
                created_at=bundle.meeting.occurred_at,
                due_date=item.due_date,
                state="COMMITTED" if owner_id or item.owner else "PROPOSED",
                mentions=[mention_of(item, bundle, owner_name)],
            )
            tracker.start(c, bundle)
            c.timeline[bundle.meeting_id] = c.state
            created.append(c)

        for c in open_before:
            obs = observations[c.commitment_id]
            tracker.apply(c, bundle, obs)
            tracker.end_meeting(c, bundle, mentioned=obs.mentioned)
        for c in existing:
            if c not in open_before:
                c.timeline[bundle.meeting_id] = c.state
        commitments.extend(created)
    return commitments
