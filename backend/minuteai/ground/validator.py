"""GroundingValidator — deterministic verification of LLM-extracted artifacts.

Any artifact carrying ``evidence: list[EvidenceSpan]`` is checked against the
transcript(s) it cites. Checks, in order:

  1. existence   — every cited utt_id exists                     UTT_NOT_FOUND
  2. quote       — each quote appears in its utterance            QUOTE_NOT_FOUND
                   (normalized; fuzzy partial ratio >= 0.85)
  3. timestamp   — span start/end within the utterance ± 0.5 s    TIMESTAMP_MISMATCH
  4. speaker     — named owner/participants (and each span's own  SPEAKER_MISMATCH
                   label) match the cited utterances' speakers
  5. entity      — names, numbers, dates and technical terms in   ENTITY_NOT_SUPPORTED
                   the artifact's text appear in the cited text
                   (relative dates may resolve to stated dates)
  6. non_empty   — at least one evidence span                     NO_EVIDENCE

Every check returns a score in [0, 1] (fraction of spans / entities that
pass) so confidence degrades smoothly; a check *fails* when its score < 1.
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from rapidfuzz import fuzz

from minuteai.extract.dates import relative_phrases, resolve_relative_date
from minuteai.extract.speakers import name_speakers
from minuteai.models.artifacts import ActionItem, Decision, GroundedSentence
from minuteai.models.evidence import EvidenceSpan
from minuteai.models.grounding import CheckFailure, CheckName, FailReason, ValidationResult
from minuteai.models.transcript import Transcript, Utterance

log = logging.getLogger(__name__)

QUOTE_THRESHOLD = 0.85
TIMESTAMP_TOLERANCE = 0.5

# Severity weights used for the confidence score.
WEIGHTS: dict[str, float] = {
    "existence": 3.0, "quote_fidelity": 3.0, "timestamp": 1.0,
    "speaker": 1.5, "entity": 2.0, "non_empty": 3.0,
}
CHECK_ORDER: tuple[CheckName, ...] = (
    "existence", "quote_fidelity", "timestamp", "speaker", "entity", "non_empty",
)

_PUNCT = str.maketrans({c: " " for c in string.punctuation if c not in "'%"})
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "fifteen": "15", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "hundred": "100", "thousand": "1000", "half": "50%", "percent": "%",
}
# "one" is too often a pronoun ("the old one"), so it is not treated as a number entity.
_ENTITY_NUMBER_WORDS = set(NUMBER_WORDS) - {"one", "percent", "hundred", "thousand", "half"}
DATE_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "today", "tomorrow", "yesterday",
}
_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
           r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
DATE_EXPR = re.compile(
    rf"\b(?:{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTHS}(?:,?\s+\d{{4}})?"
    rf"|\d{{4}}-\d{{2}}-\d{{2}})\b",
    re.IGNORECASE,
)
COMMON_CAPITALIZED = {
    "i", "the", "a", "an", "they", "we", "it", "this", "that", "these", "those", "there",
    "he", "she", "his", "her", "their", "our", "team", "everyone", "someone", "nobody",
    "speaker", "unknown", "ok", "okay", "also", "however", "next", "then", "after", "before",
}


def normalize(text: str) -> str:
    """lowercase, strip punctuation (keep apostrophes/%), collapse whitespace, digits for words."""
    text = text.lower().replace("’", "'").translate(_PUNCT)
    text = re.sub(r"'s\b", "", text).replace("'", "")
    tokens = [NUMBER_WORDS.get(t, t) for t in text.split()]
    return " ".join(tokens)


@lru_cache(maxsize=1)
def _spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except (ImportError, OSError):
        return None


SPACY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "DATE", "TIME", "MONEY", "CARDINAL",
                "PERCENT", "QUANTITY", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC", "LOC"}


@dataclass
class CheckOutcome:
    score: float = 1.0
    failures: list[CheckFailure] = field(default_factory=list)


def artifact_text(artifact) -> str:
    if isinstance(artifact, GroundedSentence):
        return artifact.text
    if isinstance(artifact, Decision):
        return ". ".join(filter(None, [artifact.statement, artifact.rationale]))
    if isinstance(artifact, ActionItem):
        return ". ".join(filter(None, [artifact.description, artifact.due_raw]))
    return getattr(artifact, "text", "")


class GroundingValidator:
    def __init__(
        self,
        transcripts: Transcript | dict[str, Transcript],
        quote_threshold: float = QUOTE_THRESHOLD,
        timestamp_tolerance: float = TIMESTAMP_TOLERANCE,
        reference_date: date | None = None,
        use_spacy: bool = True,
    ) -> None:
        if isinstance(transcripts, Transcript):
            self.default = self._named(transcripts)
            self.transcripts = {transcripts.meeting_id: self.default}
        else:
            self.transcripts = {k: self._named(t) for k, t in transcripts.items()}
            self.default = next(iter(self.transcripts.values()), None)
        self.quote_threshold = quote_threshold
        self.ts_tol = timestamp_tolerance
        self.reference_date = reference_date
        self.use_spacy = use_spacy
        self._proper_vocab = self._build_proper_vocab()
        self._name_to_labels = self._build_name_index()

    @staticmethod
    def _named(t: Transcript) -> Transcript:
        if any(u.speaker_name for u in t.utterances):
            return t
        return name_speakers(t)

    def _build_proper_vocab(self) -> set[str]:
        """Words written capitalized mid-sentence anywhere in the transcripts."""
        vocab: set[str] = set()
        for t in self.transcripts.values():
            for u in t.utterances:
                for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][\w-]*)", u.text):
                    vocab.add(m.group(1).lower())
                vocab.update(n.lower() for n in [u.speaker_name] if n)
        return vocab - COMMON_CAPITALIZED

    def _build_name_index(self) -> dict[str, set[str]]:
        idx: dict[str, set[str]] = {}
        for t in self.transcripts.values():
            for u in t.utterances:
                if u.speaker_name:
                    idx.setdefault(u.speaker_name.lower(), set()).add(u.speaker_label)
        return idx

    # -- lookup ----------------------------------------------------------------
    def utterance(self, span: EvidenceSpan) -> Utterance | None:
        t = self.transcripts.get(span.meeting_id) if span.meeting_id else self.default
        return t.get(span.utt_id) if t else None

    # -- individual checks -------------------------------------------------------
    def check_existence(self, spans: list[EvidenceSpan]) -> CheckOutcome:
        missing = [s for s in spans if self.utterance(s) is None]
        if not spans or not missing:
            return CheckOutcome()
        detail = ", ".join(
            f"{s.meeting_id + ':' if s.meeting_id else ''}{s.utt_id}" for s in missing
        )
        return CheckOutcome(
            1 - len(missing) / len(spans),
            [CheckFailure(check="existence", reason="UTT_NOT_FOUND", detail=f"utt {detail}")],
        )

    def quote_matches(self, quote: str, utterance_text: str) -> float:
        q, t = normalize(quote), normalize(utterance_text)
        if not q:
            return 0.0
        if q in t:
            return 1.0
        return fuzz.partial_ratio(q, t) / 100

    def check_quote_fidelity(self, spans: list[EvidenceSpan]) -> CheckOutcome:
        bad: list[str] = []
        checked = 0
        for s in spans:
            u = self.utterance(s)
            if u is None:
                continue  # already reported by existence
            checked += 1
            ratio = self.quote_matches(s.quote, u.text)
            if ratio < self.quote_threshold:
                bad.append(f"utt {s.utt_id}: “{s.quote[:80]}” (match {ratio:.2f})")
        if not bad:
            return CheckOutcome()
        return CheckOutcome(
            1 - len(bad) / max(checked, 1),
            [CheckFailure(check="quote_fidelity", reason="QUOTE_NOT_FOUND",
                          detail="; ".join(bad))],
        )

    def check_timestamps(self, spans: list[EvidenceSpan]) -> CheckOutcome:
        bad: list[str] = []
        checked = 0
        for s in spans:
            u = self.utterance(s)
            if u is None:
                continue
            checked += 1
            ok = (
                s.start_sec <= s.end_sec + 1e-6
                and s.start_sec >= u.start - self.ts_tol
                and s.end_sec <= u.end + self.ts_tol
            )
            if not ok:
                bad.append(f"utt {s.utt_id}: span {s.start_sec:.2f}–{s.end_sec:.2f}s vs "
                           f"utterance {u.start:.2f}–{u.end:.2f}s")
        if not bad:
            return CheckOutcome()
        return CheckOutcome(
            1 - len(bad) / max(checked, 1),
            [CheckFailure(check="timestamp", reason="TIMESTAMP_MISMATCH", detail="; ".join(bad))],
        )

    def named_speakers(self, artifact) -> list[str]:
        """Speaker labels the artifact attributes something to."""
        names: list[str] = []
        if isinstance(artifact, ActionItem) and artifact.owner:
            names.append(artifact.owner_speaker_label or artifact.owner)
        elif isinstance(artifact, Decision):
            names.extend(artifact.participants)
        labels: list[str] = []
        for n in names:
            if n.startswith("SPEAKER_") or n == "UNKNOWN":
                labels.append(n)
            elif n.lower() in self._name_to_labels:
                labels.append("|".join(sorted(self._name_to_labels[n.lower()])))
            else:
                labels.append(f"?{n}")  # a name no speaker in the transcript has
        return labels

    def check_speakers(self, artifact, spans: list[EvidenceSpan]) -> CheckOutcome:
        utts = [(s, self.utterance(s)) for s in spans]
        present = {u.speaker_label for _, u in utts if u}
        bad: list[str] = []
        for s, u in utts:
            if u and s.speaker_label != u.speaker_label:
                bad.append(f"utt {s.utt_id} labelled {s.speaker_label}, spoken by "
                           f"{u.speaker_label}")
        named = self.named_speakers(artifact)
        for label in named:
            options = set(label.split("|"))
            if label.startswith("?") or not options & present:
                shown = label.lstrip("?")
                bad.append(f"{shown} not among cited speakers {sorted(present)}")
        total = len(spans) + len(named)
        if not bad:
            return CheckOutcome()
        return CheckOutcome(
            max(0.0, 1 - len(bad) / max(total, 1)),
            [CheckFailure(check="speaker", reason="SPEAKER_MISMATCH", detail="; ".join(bad))],
        )

    def extract_entities(self, text: str) -> list[str]:
        """Names, numbers, dates and technical terms that must be grounded."""
        # Whole date expressions first, so "October 30" is one entity, not two.
        ents: list[str] = [m.group(0) for m in DATE_EXPR.finditer(text)]
        text = DATE_EXPR.sub(" ", text)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sent in sentences:
            tokens = re.findall(r"[A-Za-z][\w'’-]*|\d[\d.,:%]*\w*", sent)
            for i, tok in enumerate(tokens):
                clean = re.sub(r"['’]s$", "", tok)
                low = clean.lower()
                if re.match(r"\d", clean) or re.search(r"\d", clean):
                    ents.append(clean.rstrip(".,"))  # 800, p95, 5432, 3pm, 30%
                elif low in _ENTITY_NUMBER_WORDS or low in DATE_WORDS:
                    ents.append(low)
                elif clean[0].isupper() and low not in COMMON_CAPITALIZED:
                    if i > 0 or low in self._proper_vocab:
                        ents.append(clean)
                elif re.search(r"[a-z][A-Z]|[A-Z]{2,}", clean):  # camelCase / acronyms
                    ents.append(clean)
        if self.use_spacy and (nlp := _spacy()) is not None:
            for ent in nlp(text).ents:
                if ent.label_ in SPACY_LABELS and ent.label_ not in ("DATE", "TIME", "CARDINAL"):
                    ents.append(ent.text)
        seen: dict[str, None] = {}
        for e in ents:
            seen.setdefault(e, None)
        return list(seen)

    def _supported(self, entity: str, corpus: str, phrases: list[str]) -> bool:
        e = normalize(entity)
        if not e:
            return True
        if re.search(rf"(?<!\w){re.escape(e)}(?!\w)", corpus):
            return True
        if e.endswith("s") and re.search(rf"(?<!\w){re.escape(e[:-1])}(?!\w)", corpus):
            return True
        # resolved forms of relative date phrases ("Friday" → "October 30")
        if self.reference_date and (e in DATE_WORDS or re.search(r"\d", e)):
            stated = resolve_relative_date(entity, self.reference_date)
            if stated and any(resolve_relative_date(p, self.reference_date) == stated
                              for p in phrases):
                return True
        return False

    def check_entities(self, artifact, spans: list[EvidenceSpan]) -> CheckOutcome:
        utts = [u for s in spans if (u := self.utterance(s))]
        raw = " ".join(u.text for u in utts)
        names = " ".join(u.speaker_name or "" for u in utts)
        if isinstance(artifact, ActionItem) and artifact.owner:
            # an owner resolved via speaker naming is grounded through check 4
            names += f" {artifact.owner}" if artifact.owner_speaker_label else ""
        corpus = normalize(f"{raw} {names}")
        phrases = relative_phrases(raw)
        entities = self.extract_entities(artifact_text(artifact))
        missing = [e for e in entities if not self._supported(e, corpus, phrases)]
        if not missing:
            return CheckOutcome()
        return CheckOutcome(
            1 - len(missing) / len(entities),
            [CheckFailure(check="entity", reason="ENTITY_NOT_SUPPORTED",
                          detail="unsupported: " + ", ".join(missing))],
        )

    @staticmethod
    def check_non_empty(spans: list[EvidenceSpan]) -> CheckOutcome:
        if spans:
            return CheckOutcome()
        return CheckOutcome(0.0, [CheckFailure(check="non_empty", reason="NO_EVIDENCE",
                                               detail="artifact cites no evidence")])

    # -- orchestration -----------------------------------------------------------
    def validate(self, artifact) -> ValidationResult:
        spans: list[EvidenceSpan] = list(getattr(artifact, "evidence", []) or [])
        if not spans:
            outcome = self.check_non_empty(spans)
            return ValidationResult(passed=False, failures=outcome.failures, confidence=0.0)
        outcomes: dict[str, CheckOutcome] = {
            "existence": self.check_existence(spans),
            "quote_fidelity": self.check_quote_fidelity(spans),
            "timestamp": self.check_timestamps(spans),
            "speaker": self.check_speakers(artifact, spans),
            "entity": self.check_entities(artifact, spans),
            "non_empty": self.check_non_empty(spans),
        }
        failures = [f for name in CHECK_ORDER for f in outcomes[name].failures]
        total = sum(WEIGHTS.values())
        confidence = sum(WEIGHTS[n] * max(0.0, o.score) for n, o in outcomes.items()) / total
        return ValidationResult(
            passed=not failures, failures=failures, confidence=round(confidence, 4)
        )

    def reasons(self, artifact) -> set[FailReason]:
        return {f.reason for f in self.validate(artifact).failures}
