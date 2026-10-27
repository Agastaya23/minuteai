"""The built-in synthetic sample series, with hand-written gold annotations.

Lets the whole harness run offline (with --replay) as a smoke test of every
metric, and gives a tiny fully-labelled set for commitment-lifecycle accuracy.
"""

from __future__ import annotations

from eval.common import EvalMeeting
from minuteai.samples import SAMPLE_MEETINGS, SAMPLE_SERIES_ID, sample_transcript

GOLD = {
    "demo-sprint-planning": {
        "abstract": [
            "The team agreed to move the payments service from MySQL to Postgres 16 for its "
            "JSON support.",
            "Rahul will write the migration script by Friday and Ananya will set up staging.",
            "Checkout latency is 800 milliseconds at p95, mostly the synchronous fraud check; "
            "the decision on making it asynchronous was deferred.",
            "Priya will update the on-call rotation and the Thursday demo stays.",
        ],
        "actions": [
            ("Write the migration script by Friday", "Rahul"),
            ("Set up the staging database on port 5432 by Wednesday", "Ananya"),
            ("Ask the fraud team for their latency budget", None),
            ("Update the on-call rotation", "Priya"),
        ],
        "decisions": ["Move the payments service to Postgres 16",
                      "Defer the asynchronous fraud check decision",
                      "Keep the demo on Thursday at 3 pm"],
    },
    "demo-sprint-review": {
        "abstract": [
            "The migration script is finished and merged.",
            "Staging slipped to next Monday.",
            "The fraud team's latency budget is 200 milliseconds so the fraud check will be made "
            "asynchronous this sprint, owned by Ananya.",
            "Payments stays on Postgres despite analytics preferring MySQL; Rahul will write a "
            "rollback plan.",
        ],
        "actions": [
            ("Finish the staging database by next Monday", "Ananya"),
            ("Make the fraud check asynchronous", "Ananya"),
            ("Write the rollback plan by Wednesday", "Rahul"),
        ],
        "decisions": ["Make the fraud check asynchronous this sprint",
                      "Payments stays on Postgres"],
    },
    "demo-sprint-retro": {
        "abstract": [
            "Staging went live on Monday and the rollback plan is done.",
            "The asynchronous fraud check is half done and expected next week.",
            "Thursday demos are cancelled.",
        ],
        "actions": [
            ("Finish the asynchronous fraud check next week", "Ananya"),
            ("Send the release notes to the support team tomorrow", "Rahul"),
        ],
        "decisions": ["Cancel the Thursday demos"],
    },
}

# Gold lifecycle state per commitment (by its originating meeting + description).
GOLD_COMMITMENT_STATES = {
    ("demo-sprint-planning", "migration script"): "FULFILLED",
    ("demo-sprint-planning", "staging database"): "FULFILLED",
    ("demo-sprint-planning", "latency budget"): "FULFILLED",
    ("demo-sprint-planning", "on-call rotation"): "SILENTLY_DROPPED",
    ("demo-sprint-review", "asynchronous fraud check"): "SLIPPED",
    ("demo-sprint-review", "rollback plan"): "FULFILLED",
    ("demo-sprint-retro", "release notes"): "COMMITTED",
}


def load(n: int | None = None) -> list[EvalMeeting]:
    out = []
    for mid in list(SAMPLE_MEETINGS)[:n]:
        g = GOLD[mid]
        out.append(EvalMeeting(
            meeting_id=mid, transcript=sample_transcript(mid),
            gold_abstract=g["abstract"], gold_actions=[a for a, _ in g["actions"]],
            gold_decisions=g["decisions"],
            gold_action_owners={i: o for i, (_, o) in enumerate(g["actions"])},
            series_id=SAMPLE_SERIES_ID, source="samples",
        ))
    return out
