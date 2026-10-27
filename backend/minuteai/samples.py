"""Deterministic synthetic meetings.

Used by the test-suite fixtures, the seed script (so a fresh clone has
something to show), and offline evaluation runs. Word timings are generated
at a fixed speaking rate so they are monotonic and realistic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from minuteai.models.transcript import Transcript, Utterance, Word

WORDS_PER_SEC = 2.6
GAP_SEC = 0.6

# (speaker_label, text). Speaker names are introduced in-dialogue so entity
# resolution has something to find.
SPRINT_PLANNING: list[tuple[str, str]] = [
    ("SPEAKER_00", "Hi everyone, I'm Priya. Let's start the sprint planning for the payments service."),
    ("SPEAKER_01", "Thanks Priya. Rahul here. The main blocker is the database migration."),
    ("SPEAKER_02", "Ananya here. I looked at the options and Postgres 16 looks like the best fit."),
    ("SPEAKER_00", "What about keeping MySQL? Would that be cheaper for us?"),
    ("SPEAKER_02", "MySQL would cost about the same, but Postgres gives us better JSON support."),
    ("SPEAKER_01", "I agree with Ananya. Postgres also matches what the analytics team uses."),
    ("SPEAKER_00", "Okay, so we are agreed. We will move the payments service to Postgres 16."),
    ("SPEAKER_01", "Great. I will write the migration script by Friday."),
    ("SPEAKER_00", "Thanks Rahul. Ananya, can you set up the staging database on port 5432?"),
    ("SPEAKER_02", "Sure, I can have staging ready by Wednesday."),
    ("SPEAKER_00", "Next topic is the checkout latency. We are seeing around 800 milliseconds at p95."),
    ("SPEAKER_01", "Most of that is the fraud check call. It is synchronous right now."),
    ("SPEAKER_02", "We could make the fraud check asynchronous and cache the risk scores."),
    ("SPEAKER_00", "Let's not decide that today. We need numbers from the fraud team first."),
    ("SPEAKER_01", "Someone should ask the fraud team for their latency budget."),
    ("SPEAKER_00", "Right, let's park that until next week."),
    ("SPEAKER_02", "One more thing, the on-call rotation needs an update for December."),
    ("SPEAKER_00", "I will update the on-call rotation before the end of the month."),
    ("SPEAKER_01", "Sounds good. Should we keep the demo on Thursday?"),
    ("SPEAKER_00", "Yes, the demo stays on Thursday at 3 pm. Thanks everyone."),
]

SPRINT_REVIEW: list[tuple[str, str]] = [
    ("SPEAKER_00", "Hi, Priya here again. Let's review what happened since last sprint planning."),
    ("SPEAKER_01", "This is Rahul. I finished the migration script for Postgres, it is merged."),
    ("SPEAKER_02", "Ananya here. Staging is not ready yet, I need until next Monday."),
    ("SPEAKER_00", "Okay, so staging slips to Monday. That is fine."),
    ("SPEAKER_01", "I talked to the fraud team. Their latency budget is 200 milliseconds."),
    ("SPEAKER_02", "Then making the fraud check asynchronous is definitely worth it."),
    ("SPEAKER_00", "Agreed, we will make the fraud check asynchronous this sprint."),
    ("SPEAKER_02", "I can own the asynchronous fraud check work."),
    ("SPEAKER_00", "Thanks Ananya. Anything else?"),
    ("SPEAKER_01", "Actually, the analytics team now wants to stay on MySQL."),
    ("SPEAKER_00", "That does not change our decision. Payments stays on Postgres."),
    ("SPEAKER_01", "Fine. I will write the rollback plan by Wednesday."),
]

SPRINT_RETRO: list[tuple[str, str]] = [
    ("SPEAKER_00", "Priya here. Quick retro on the payments work."),
    ("SPEAKER_02", "Ananya here. Staging is done, it went live on Monday as planned."),
    ("SPEAKER_01", "Rahul here. The rollback plan is written and reviewed."),
    ("SPEAKER_00", "Great. The asynchronous fraud check is still in progress, right?"),
    ("SPEAKER_02", "Yes, it is about half done. I expect to finish it next week."),
    ("SPEAKER_00", "We decided to cancel the Thursday demos going forward, they take too long."),
    ("SPEAKER_01", "I will send the release notes to the support team tomorrow."),
    ("SPEAKER_00", "Thanks. Let's wrap up."),
]

SAMPLE_MEETINGS: dict[str, dict] = {
    "demo-sprint-planning": {
        "title": "Payments sprint planning",
        "lines": SPRINT_PLANNING,
        "days_ago": 14,
    },
    "demo-sprint-review": {
        "title": "Payments sprint review",
        "lines": SPRINT_REVIEW,
        "days_ago": 7,
    },
    "demo-sprint-retro": {
        "title": "Payments retro",
        "lines": SPRINT_RETRO,
        "days_ago": 1,
    },
}
SAMPLE_SERIES_ID = "payments-weekly"
SAMPLE_PARTICIPANTS = ["Priya", "Rahul", "Ananya"]


def build_transcript(meeting_id: str, lines: list[tuple[str, str]]) -> Transcript:
    utterances: list[Utterance] = []
    t = 0.0
    for i, (speaker, text) in enumerate(lines):
        words: list[Word] = []
        start = t
        for token in text.split():
            dur = max(0.15, len(token) / (WORDS_PER_SEC * 5))
            words.append(Word(text=token, start=round(t, 3), end=round(t + dur, 3)))
            t += 1.0 / WORDS_PER_SEC
        end = words[-1].end
        utterances.append(
            Utterance(
                utt_id=i, speaker_label=speaker, text=text, start=round(start, 3),
                end=end, words=words,
            )
        )
        t = end + GAP_SEC
    return Transcript(
        meeting_id=meeting_id,
        utterances=utterances,
        duration_sec=round(t, 3),
        backend="synthetic",
        model_version="synthetic-v1",
    )


def sample_transcript(meeting_id: str = "demo-sprint-planning") -> Transcript:
    return build_transcript(meeting_id, SAMPLE_MEETINGS[meeting_id]["lines"])


def sample_occurred_at(meeting_id: str, now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return (now - timedelta(days=SAMPLE_MEETINGS[meeting_id]["days_ago"])).replace(
        hour=10, minute=0, second=0, microsecond=0
    )
