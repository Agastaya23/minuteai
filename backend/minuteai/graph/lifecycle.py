"""Commitment lifecycle inference.

    PROPOSED → COMMITTED → { FULFILLED | SLIPPED | SILENTLY_DROPPED | REVERSED }
                              SLIPPED → { FULFILLED | SLIPPED | SILENTLY_DROPPED | REVERSED }

State is inferred from *later* meetings; every transition carries evidence
spans from the meeting that caused it:
  FULFILLED        a later utterance about it states completion
  SLIPPED          re-raised with a new deadline, or an explicit delay
  SILENTLY_DROPPED not mentioned in the next N meetings (default 2) and past due
  REVERSED         a later decision explicitly cancels/contradicts it

Detection is deterministic (similarity + cue phrases). An optional LLM judge
can be supplied for mentions the cues cannot classify; its evidence is
validated by the GroundingValidator before it is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from minuteai.extract import prompts
from minuteai.extract.chunker import render_utterances
from minuteai.extract.llm_client import LLMClient, LLMError
from minuteai.extract.schemas import LLMCommitmentStatus
from minuteai.graph.entity_resolution import MeetingBundle
from minuteai.graph.similarity import Similarity, is_completion, is_delay, is_reversal
from minuteai.models.evidence import EvidenceSpan
from minuteai.models.graph import TERMINAL_STATES, Commitment, CommitmentState, Transition
from minuteai.models.transcript import Utterance

MENTION_THRESHOLD = 0.5
REVERSAL_THRESHOLD = 0.55
JUDGE_THRESHOLD = 0.35
DEFAULT_SILENCE_MEETINGS = 2


def span_of(u: Utterance, meeting_id: str, quote: str | None = None) -> EvidenceSpan:
    return EvidenceSpan(utt_id=u.utt_id, start_sec=u.start, end_sec=u.end,
                        speaker_label=u.speaker_label, quote=(quote or u.text)[:200],
                        meeting_id=meeting_id)


@dataclass
class Observation:
    to_state: CommitmentState | None
    reason: str
    evidence: list[EvidenceSpan] = field(default_factory=list)
    mentioned: bool = False


PRIORITY = {"REVERSED": 3, "FULFILLED": 2, "SLIPPED": 1, None: 0}


class LifecycleTracker:
    def __init__(
        self,
        sim: Similarity,
        silence_meetings: int = DEFAULT_SILENCE_MEETINGS,
        judge: LLMClient | None = None,
    ) -> None:
        self.sim = sim
        self.silence_meetings = silence_meetings
        self.judge = judge
        self.silent: dict[str, int] = {}

    # -- transitions ---------------------------------------------------------
    @staticmethod
    def transition(c: Commitment, to: CommitmentState, bundle: MeetingBundle, reason: str,
                   evidence: list[EvidenceSpan]) -> None:
        c.transitions.append(Transition(
            commitment_id=c.commitment_id, from_state=c.state if c.transitions else None,
            to_state=to, meeting_id=bundle.meeting_id, occurred_at=bundle.meeting.occurred_at,
            reason=reason, evidence=evidence,
        ))
        c.state = to

    def start(self, c: Commitment, bundle: MeetingBundle) -> None:
        reason = "Committed with an owner" if c.state == "COMMITTED" else \
            "Raised without a clear owner"
        ev = c.mentions[0].evidence
        c.transitions.append(Transition(
            commitment_id=c.commitment_id, from_state=None, to_state=c.state,
            meeting_id=bundle.meeting_id, occurred_at=bundle.meeting.occurred_at,
            reason=reason, evidence=ev,
        ))
        self.silent[c.commitment_id] = 0

    # -- observation of a later meeting ---------------------------------------
    def _texts(self, c: Commitment) -> list[str]:
        return [m.description for m in c.mentions]

    def observe(self, c: Commitment, bundle: MeetingBundle) -> Observation:
        """What does this (later) meeting's dialogue and decisions say about `c`?"""
        best = Observation(None, "")
        texts = self._texts(c)
        judge_candidates: list[Utterance] = []
        for u in bundle.transcript.utterances:
            score = max(self.sim.score(t, u.text) for t in texts)
            if score >= JUDGE_THRESHOLD:
                judge_candidates.append(u)
            if score < MENTION_THRESHOLD:
                continue
            best.mentioned = True
            if is_completion(u.text):
                obs = Observation("FULFILLED", f"Completion stated by {u.speaker}: “{u.text}”",
                                  [span_of(u, bundle.meeting_id)], True)
            elif is_delay(u.text):
                obs = Observation("SLIPPED", f"Delay stated by {u.speaker}: “{u.text}”",
                                  [span_of(u, bundle.meeting_id)], True)
            else:
                continue
            if PRIORITY[obs.to_state] > PRIORITY[best.to_state]:
                best = obs

        if bundle.summary:
            for d in bundle.summary.decisions:
                if not (d.status == "reversed" or is_reversal(d.statement)):
                    continue
                score = max(self.sim.score(t, d.statement) for t in texts)
                if score >= REVERSAL_THRESHOLD:
                    ev = [e.model_copy(update={"meeting_id": bundle.meeting_id})
                          for e in d.evidence]
                    return Observation("REVERSED", f"Reversed by decision: “{d.statement}”",
                                       ev, True)

        if best.to_state is None and self.judge is not None and judge_candidates:
            judged = self._ask_judge(c, bundle, judge_candidates)
            if judged is not None:
                return judged
        return best

    def _ask_judge(self, c: Commitment, bundle: MeetingBundle, utts: list[Utterance]
                   ) -> Observation | None:
        from minuteai.ground.validator import GroundingValidator
        from minuteai.models.artifacts import GroundedSentence

        try:
            resp = self.judge.complete_structured(
                prompts.render(
                    prompts.COMMITMENT_STATUS_PROMPT, commitment=c.description,
                    owner=c.owner_name or "unassigned", due=c.due_date or "none",
                    transcript=render_utterances(utts),
                ),
                LLMCommitmentStatus, purpose="lifecycle", meeting_id=bundle.meeting_id,
            )
        except LLMError:
            return None
        state = {"fulfilled": "FULFILLED", "slipped": "SLIPPED", "reversed": "REVERSED"}.get(
            resp.status)
        if state is None:
            return Observation(None, "", mentioned=resp.status == "in_progress")
        ev = [span_of(u, bundle.meeting_id, e.quote) for e in resp.evidence
              if (u := bundle.transcript.get(e.utt_id)) is not None]
        claim = GroundedSentence(text=c.description, evidence=ev)
        validator = GroundingValidator({bundle.meeting_id: bundle.transcript}, use_spacy=False)
        result = validator.validate(claim)
        if not ev or {"UTT_NOT_FOUND", "QUOTE_NOT_FOUND"} & result.reasons:
            return None
        return Observation(state, f"LLM judge: {resp.status}", ev, True)

    # -- a linked recurrence (action item re-raised) ----------------------------
    def on_recurrence(self, c: Commitment, bundle: MeetingBundle) -> Observation:
        m = c.mentions[-1]
        quote_text = " ".join(e.quote for e in m.evidence)
        new_due = m.due_date
        if new_due and new_due != c.due_date:
            reason = f"Re-raised with a new deadline “{m.due_raw}” ({new_due.isoformat()})"
            if c.due_date:
                reason += f", previously {c.due_date.isoformat()}"
            return Observation("SLIPPED", reason, m.evidence, True)
        if is_delay(quote_text) or is_delay(m.description):
            return Observation("SLIPPED", f"Re-raised with a delay: “{quote_text}”",
                               m.evidence, True)
        if is_completion(quote_text):
            return Observation("FULFILLED", f"Reported complete: “{quote_text}”", m.evidence, True)
        return Observation(None, "Re-raised", m.evidence, True)

    # -- end of a meeting: apply, count silence, record timeline ----------------
    def apply(self, c: Commitment, bundle: MeetingBundle, obs: Observation) -> None:
        if obs.mentioned:
            self.silent[c.commitment_id] = 0
        if obs.to_state and c.state not in TERMINAL_STATES:
            self.transition(c, obs.to_state, bundle, obs.reason, obs.evidence)
            if obs.to_state == "SLIPPED" and c.mentions[-1].meeting_id == bundle.meeting_id \
                    and c.mentions[-1].due_date:
                c.due_date = c.mentions[-1].due_date

    def end_meeting(self, c: Commitment, bundle: MeetingBundle, mentioned: bool) -> None:
        if c.state not in TERMINAL_STATES and not mentioned:
            n = self.silent.get(c.commitment_id, 0) + 1
            self.silent[c.commitment_id] = n
            day: date = bundle.meeting.occurred_at.date()
            if n >= self.silence_meetings and c.due_date and c.due_date < day:
                last = c.mentions[-1]
                self.transition(
                    c, "SILENTLY_DROPPED", bundle,
                    f"Not mentioned in {n} meetings since {last.occurred_at.date().isoformat()} "
                    f"and past its due date {c.due_date.isoformat()}",
                    last.evidence,
                )
        c.timeline[bundle.meeting_id] = c.state
