"""Deterministic synthetic meetings.

Used by the test-suite fixtures, the seed script (so a fresh clone has
something to show), and offline evaluation runs. Word timings are generated
at a fixed speaking rate so they are monotonic and realistic.
"""

from __future__ import annotations

import json
import re
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
    ("SPEAKER_00", "I will update the on-call rotation by next Monday."),
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


def build_transcript(
    meeting_id: str,
    lines: list[tuple[str, str]],
    timings: list[tuple[float, float]] | None = None,
) -> Transcript:
    """Synthetic transcript; with `timings` (per-utterance start/end, e.g. measured
    from generated sample audio) words are spread across each real clip."""
    if timings is not None:
        return _timed_transcript(meeting_id, lines, timings)
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


def _timed_transcript(
    meeting_id: str, lines: list[tuple[str, str]], timings: list[tuple[float, float]]
) -> Transcript:
    utterances: list[Utterance] = []
    for i, ((speaker, text), (start, end)) in enumerate(zip(lines, timings, strict=True)):
        tokens = text.split()
        weights = [len(tok) + 2 for tok in tokens]
        total = sum(weights)
        words, t = [], start
        for tok, w in zip(tokens, weights, strict=True):
            dur = (end - start) * w / total
            words.append(Word(text=tok, start=round(t, 3), end=round(t + dur * 0.9, 3)))
            t += dur
        utterances.append(Utterance(utt_id=i, speaker_label=speaker, text=text,
                                    start=round(start, 3), end=round(end, 3), words=words))
    return Transcript(meeting_id=meeting_id, utterances=utterances,
                      duration_sec=round(timings[-1][1] + GAP_SEC, 3), backend="synthetic",
                      model_version="synthetic-v1+tts-aligned")


def sample_transcript(
    meeting_id: str = "demo-sprint-planning", timings: list[tuple[float, float]] | None = None
) -> Transcript:
    return build_transcript(meeting_id, SAMPLE_MEETINGS[meeting_id]["lines"], timings)


def sample_occurred_at(meeting_id: str, now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return (now - timedelta(days=SAMPLE_MEETINGS[meeting_id]["days_ago"])).replace(
        hour=10, minute=0, second=0, microsecond=0
    )


# ---------------------------------------------------------------------------
# Recorded LLM responses for the sample meetings.
#
# They let the seed script and offline demos run the *real* extraction
# pipeline (evidence attachment, merging, due-date resolution, grounding
# filter) without an API key. A few artifacts are deliberately ungrounded —
# a non-existent utterance, a fabricated quote, a wrong speaker, an
# unsupported number — so the dropped-artifacts view has something to show.
# ---------------------------------------------------------------------------


def _ev(*pairs) -> list[dict]:
    return [{"utt_id": u, "quote": q} for u, q in pairs]


SAMPLE_LLM_RESPONSES: dict[str, dict[str, dict]] = {
    "demo-sprint-planning": {
        "LLMSummaryResponse": {"sentences": [
            {"text": "The team agreed to move the payments service to Postgres 16 because it "
                     "offers better JSON support.",
             "evidence": _ev((4, "Postgres gives us better JSON support"),
                             (6, "We will move the payments service to Postgres 16"))},
            {"text": "Rahul will write the migration script and Ananya will set up the staging "
                     "database.",
             "evidence": _ev((7, "I will write the migration script by Friday"),
                             (8, "can you set up the staging database on port 5432"),
                             (9, "I can have staging ready by Wednesday"))},
            {"text": "Checkout latency is around 800 milliseconds at p95, mostly because the "
                     "fraud check is synchronous.",
             "evidence": _ev((10, "We are seeing around 800 milliseconds at p95"),
                             (11, "Most of that is the fraud check call. It is synchronous"))},
            {"text": "Moving to Postgres is expected to cut database costs by 30 percent.",
             "evidence": _ev((4, "MySQL would cost about the same"))},
            {"text": "The team also agreed to hire two more backend engineers.",
             "evidence": _ev((42, "we should hire two more backend engineers"))},
            {"text": "The demo stays on Thursday at 3 pm.",
             "evidence": _ev((19, "the demo stays on Thursday at 3 pm"))},
        ]},
        "LLMDecisionsResponse": {"decisions": [
            {"statement": "Move the payments service to Postgres 16",
             "rationale": "Better JSON support and alignment with the analytics team",
             "participants": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], "status": "agreed",
             "confidence": 0.95,
             "evidence": _ev((2, "Postgres 16 looks like the best fit"),
                             (4, "Postgres gives us better JSON support"),
                             (5, "Postgres also matches what the analytics team uses"),
                             (6, "we are agreed. We will move the payments service to Postgres 16"))},
            {"statement": "Defer the decision on making the fraud check asynchronous until the "
                          "fraud team provides numbers",
             "rationale": None, "participants": ["SPEAKER_00"], "status": "deferred",
             "confidence": 0.8,
             "evidence": _ev((13, "Let's not decide that today"),
                             (15, "let's park that until next week"))},
            {"statement": "Keep the demo on Thursday at 3 pm", "rationale": None,
             "participants": ["SPEAKER_00", "SPEAKER_01"], "status": "agreed", "confidence": 0.9,
             "evidence": _ev((18, "Should we keep the demo on Thursday?"),
                             (19, "the demo stays on Thursday at 3 pm"))},
            {"statement": "Adopt Redis to cache fraud risk scores", "rationale": None,
             "participants": ["SPEAKER_02"], "status": "agreed", "confidence": 0.6,
             "evidence": _ev((12, "we will use Redis for the risk score cache"))},
        ]},
        "LLMActionItemsResponse": {"action_items": [
            {"description": "Write the migration script", "owner": "Rahul",
             "owner_speaker_label": "SPEAKER_01", "owner_resolution_method": "self_commitment",
             "owner_confidence": 0.95, "due_raw": "by Friday", "confidence": 0.95,
             "evidence": _ev((7, "I will write the migration script by Friday"))},
            {"description": "Set up the staging database on port 5432", "owner": "Ananya",
             "owner_speaker_label": "SPEAKER_02", "owner_resolution_method": "assigned_by_other",
             "owner_confidence": 0.9, "due_raw": "by Wednesday", "confidence": 0.9,
             "evidence": _ev((8, "Ananya, can you set up the staging database on port 5432?"),
                             (9, "I can have staging ready by Wednesday"))},
            {"description": "Ask the fraud team for their latency budget", "owner": None,
             "owner_speaker_label": None, "owner_resolution_method": "unresolved",
             "owner_confidence": 0.0, "due_raw": None, "confidence": 0.8,
             "evidence": _ev((14, "Someone should ask the fraud team for their latency budget"))},
            {"description": "Update the on-call rotation for December", "owner": "Priya",
             "owner_speaker_label": "SPEAKER_00", "owner_resolution_method": "self_commitment",
             "owner_confidence": 0.9, "due_raw": "by next Monday", "confidence": 0.9,
             "evidence": _ev((16, "the on-call rotation needs an update for December"),
                             (17, "I will update the on-call rotation by next Monday"))},
            {"description": "Benchmark the asynchronous fraud check", "owner": "Rahul",
             "owner_speaker_label": "SPEAKER_01", "owner_resolution_method": "explicit_name",
             "owner_confidence": 0.5, "due_raw": None, "confidence": 0.5,
             "evidence": _ev((12, "make the fraud check asynchronous"))},
        ]},
    },
    "demo-sprint-review": {
        "LLMSummaryResponse": {"sentences": [
            {"text": "Rahul finished the Postgres migration script and it is merged.",
             "evidence": _ev((1, "I finished the migration script for Postgres, it is merged"))},
            {"text": "Staging is not ready yet and has slipped to next Monday.",
             "evidence": _ev((2, "Staging is not ready yet, I need until next Monday"),
                             (3, "staging slips to Monday"))},
            {"text": "The fraud team's latency budget is 200 milliseconds, so the team agreed to "
                     "make the fraud check asynchronous this sprint.",
             "evidence": _ev((4, "Their latency budget is 200 milliseconds"),
                             (6, "we will make the fraud check asynchronous this sprint"))},
            {"text": "Payments stays on Postgres even though the analytics team now wants MySQL.",
             "evidence": _ev((9, "the analytics team now wants to stay on MySQL"),
                             (10, "Payments stays on Postgres"))},
        ]},
        "LLMDecisionsResponse": {"decisions": [
            {"statement": "Make the fraud check asynchronous this sprint", "rationale":
                "The fraud team's latency budget is 200 milliseconds",
             "participants": ["SPEAKER_00", "SPEAKER_02"], "status": "agreed", "confidence": 0.9,
             "evidence": _ev((5, "making the fraud check asynchronous is definitely worth it"),
                             (6, "we will make the fraud check asynchronous this sprint"))},
            {"statement": "Keep the payments service on Postgres", "rationale": None,
             "participants": ["SPEAKER_00"], "status": "agreed", "confidence": 0.85,
             "evidence": _ev((10, "That does not change our decision. Payments stays on Postgres"))},
        ]},
        "LLMActionItemsResponse": {"action_items": [
            {"description": "Finish setting up the staging database", "owner": "Ananya",
             "owner_speaker_label": "SPEAKER_02", "owner_resolution_method": "self_commitment",
             "owner_confidence": 0.85, "due_raw": "until next Monday", "confidence": 0.85,
             "evidence": _ev((2, "Staging is not ready yet, I need until next Monday"))},
            {"description": "Implement the asynchronous fraud check", "owner": "Ananya",
             "owner_speaker_label": "SPEAKER_02", "owner_resolution_method": "self_commitment",
             "owner_confidence": 0.9, "due_raw": None, "confidence": 0.9,
             "evidence": _ev((7, "I can own the asynchronous fraud check work"))},
            {"description": "Write the rollback plan", "owner": "Rahul",
             "owner_speaker_label": "SPEAKER_01", "owner_resolution_method": "self_commitment",
             "owner_confidence": 0.95, "due_raw": "by Wednesday", "confidence": 0.95,
             "evidence": _ev((11, "I will write the rollback plan by Wednesday"))},
        ]},
    },
    "demo-sprint-retro": {
        "LLMSummaryResponse": {"sentences": [
            {"text": "Staging went live on Monday and the rollback plan has been written and "
                     "reviewed.",
             "evidence": _ev((1, "Staging is done, it went live on Monday"),
                             (2, "The rollback plan is written and reviewed"))},
            {"text": "The asynchronous fraud check is about half done and should finish next week.",
             "evidence": _ev((4, "it is about half done. I expect to finish it next week"))},
            {"text": "The team decided to cancel the Thursday demos going forward.",
             "evidence": _ev((5, "We decided to cancel the Thursday demos going forward"))},
        ]},
        "LLMDecisionsResponse": {"decisions": [
            {"statement": "Cancel the Thursday demos going forward",
             "rationale": "They take too long", "participants": ["SPEAKER_00"],
             "status": "reversed", "confidence": 0.9,
             "evidence": _ev((5, "We decided to cancel the Thursday demos going forward"))},
        ]},
        "LLMActionItemsResponse": {"action_items": [
            {"description": "Finish the asynchronous fraud check", "owner": "Ananya",
             "owner_speaker_label": "SPEAKER_02", "owner_resolution_method": "self_commitment",
             "owner_confidence": 0.85, "due_raw": "next week", "confidence": 0.85,
             "evidence": _ev((4, "I expect to finish it next week"))},
            {"description": "Send the release notes to the support team", "owner": "Rahul",
             "owner_speaker_label": "SPEAKER_01", "owner_resolution_method": "self_commitment",
             "owner_confidence": 0.95, "due_raw": "tomorrow", "confidence": 0.95,
             "evidence": _ev((6, "I will send the release notes to the support team tomorrow"))},
        ]},
    },
}

_TOPIC_KEYWORDS = [
    (("latency", "fraud"), "Checkout latency and the fraud check"),
    (("migration", "postgres", "mysql", "database"), "Database migration to Postgres"),
    (("on-call", "demo", "rotation"), "On-call rotation and demo schedule"),
    (("staging",), "Staging environment"),
    (("release", "retro"), "Release follow-ups"),
]


def _title_segments(prompt: str) -> dict:
    blocks = re.split(r"^Segment (\d+):$", prompt, flags=re.MULTILINE)
    topics = []
    for i in range(1, len(blocks) - 1, 2):
        text = blocks[i + 1].lower()
        title = next((t for keys, t in _TOPIC_KEYWORDS if any(k in text for k in keys)),
                     "General discussion")
        topics.append({"segment": int(blocks[i]), "title": title})
    return {"topics": topics}


class ReplayProvider:
    """LLM provider that replays SAMPLE_LLM_RESPONSES for one sample meeting."""

    name = "replay"

    def __init__(self, meeting_id: str) -> None:
        self.meeting_id = meeting_id
        self.model = "replay/sample"

    def generate(self, prompt: str, response_model):
        from minuteai.extract.llm_client import ProviderResponse

        name = response_model.__name__
        if name == "LLMTopicTitlesResponse":
            body = _title_segments(prompt)
        else:
            body = SAMPLE_LLM_RESPONSES[self.meeting_id].get(name)
            if body is None:
                body = {f: [] for f in response_model.model_fields if
                        f in ("sentences", "decisions", "action_items")}
        text = json.dumps(body)
        return ProviderResponse(text=text, input_tokens=len(prompt) // 4,
                                output_tokens=len(text) // 4, stop_reason="end_turn")
