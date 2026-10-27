"""Grounded natural-language answers over all meetings.

Retrieved entries are expanded into numbered *utterance* lines; the LLM may
only cite those numbers, which map back to real (meeting_id, utt_id) pairs.
The answer then goes through the same GroundingValidator as the minutes; an
answer that fails the drop checks is replaced, never shown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from minuteai.extract import prompts
from minuteai.extract.chunker import fmt_ts
from minuteai.extract.llm_client import LLMClient
from minuteai.extract.schemas import LLMAnswer
from minuteai.ground.filter import Strictness, log_dropped, should_drop
from minuteai.ground.validator import GroundingValidator
from minuteai.models.artifacts import GroundedSentence
from minuteai.models.evidence import EvidenceSpan
from minuteai.models.grounding import DroppedArtifact
from minuteai.models.search import SearchFilters
from minuteai.models.transcript import Transcript, Utterance
from minuteai.search.hybrid import Hit, HybridRetriever

log = logging.getLogger(__name__)

NO_RESULTS = "None of your meetings discuss this."
UNVERIFIED = ("I couldn't produce an answer that is verifiably supported by the meetings. "
              "The sources below are the closest matches.")
MAX_LINES = 60


@dataclass
class Answer:
    query: str
    answerable: bool
    sentence: GroundedSentence
    hits: list[Hit]
    flags: list[str] = field(default_factory=list)
    dropped: DroppedArtifact | None = None


class Answerer:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        transcript_loader,
        mode: Strictness | str = Strictness.BALANCED,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.load_transcript = transcript_loader  # meeting_id -> Transcript | None
        self.mode = Strictness(mode)

    def _lines(self, hits: list[Hit], transcripts: dict[str, Transcript]
               ) -> list[tuple[str, Utterance, str]]:
        """Unique (meeting_id, utterance, header) lines in retrieval order."""
        seen: set[tuple[str, int]] = set()
        lines: list[tuple[str, Utterance, str]] = []
        for h in hits:
            t = transcripts.get(h.entry.meeting_id)
            if t is None:
                continue
            date = h.entry.occurred_at.date().isoformat() if h.entry.occurred_at else "?"
            header = f"{h.entry.meeting_title} ({date})"
            for u in t.window(h.entry.start_utt, h.entry.end_utt):
                key = (h.entry.meeting_id, u.utt_id)
                if key not in seen and len(lines) < MAX_LINES:
                    seen.add(key)
                    lines.append((h.entry.meeting_id, u, header))
        return lines

    def answer(self, query: str, filters: SearchFilters | None = None, k: int = 8) -> Answer:
        hits = self.retriever.retrieve(query, filters, k)
        if not hits:
            return Answer(query, False, GroundedSentence(text=NO_RESULTS), hits)

        transcripts: dict[str, Transcript] = {}
        for h in hits:
            mid = h.entry.meeting_id
            if mid not in transcripts and (t := self.load_transcript(mid)) is not None:
                transcripts[mid] = t
        lines = self._lines(hits, transcripts)
        excerpts = "\n".join(
            f"[{n}] {header} {u.speaker} ({fmt_ts(u.start)}): {u.text}"
            for n, (_, u, header) in enumerate(lines)
        )
        resp = self.llm.complete_structured(
            prompts.render(prompts.SEARCH_ANSWER_PROMPT, query=query, excerpts=excerpts),
            LLMAnswer, purpose="search",
        )
        if not resp.answerable:
            return Answer(query, False, GroundedSentence(text=resp.answer.strip() or NO_RESULTS),
                          hits)

        evidence: list[EvidenceSpan] = []
        for ev in resp.evidence:
            if 0 <= ev.utt_id < len(lines):
                mid, u, _ = lines[ev.utt_id]
                evidence.append(EvidenceSpan(
                    utt_id=u.utt_id, start_sec=u.start, end_sec=u.end,
                    speaker_label=u.speaker_label, quote=ev.quote, meeting_id=mid,
                ))
            else:
                # cites a source number that was never shown: keep it so the
                # validator rejects it (meeting "unknown" → UTT_NOT_FOUND)
                evidence.append(EvidenceSpan(utt_id=ev.utt_id, start_sec=0, end_sec=0,
                                             speaker_label="UNKNOWN", quote=ev.quote,
                                             meeting_id="__unknown__"))
        sentence = GroundedSentence(text=resp.answer.strip(), evidence=evidence)

        validator = GroundingValidator(transcripts)
        result = validator.validate(sentence)
        if should_drop(result, self.mode):
            dropped = DroppedArtifact(
                meeting_id=",".join(sorted(transcripts)) or "search",
                artifact_type="search_answer", artifact_id=query[:80], text=sentence.text,
                evidence=evidence, failures=result.failures, confidence=result.confidence,
                mode=self.mode,
            )
            log_dropped([dropped])
            log.info("search answer dropped: %s", result.reasons)
            return Answer(query, False, GroundedSentence(text=UNVERIFIED), hits,
                          flags=sorted(result.reasons), dropped=dropped)
        sentence = sentence.model_copy(update={"validated": result.passed,
                                               "flags": sorted(result.reasons)})
        return Answer(query, True, sentence, hits, flags=sorted(result.reasons))
