"""Per-meeting speaker naming from the transcript itself (deterministic).

Signals, strongest first:
  * self-introduction  — "I'm Priya", "Rahul here", "this is Ananya", "my name is …"
  * direct address     — "Thanks Rahul" (names the previous speaker),
                          "Ananya, can you …" (names the next speaker)
  * participant list   — supplied at upload; only constrains which names are accepted

The cross-meeting identity layer (graph/entity_resolution.py) builds on this.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from minuteai.models.transcript import Transcript

_NAME = r"([A-Z][a-z]{1,20})(?![-'’\w])"
# Strong: unambiguous self-introductions.
SELF_INTRO = [
    re.compile(rf"\b(?i:I'm|I am|my name is)\s+{_NAME}"),
]
# Weak: also match ordinary speech ("Check here.", "It's Play-Doh"), so the name
# must be corroborated elsewhere in the meeting (see `addressable`).
WEAK_INTRO = [
    re.compile(rf"^(?i:(?:hi|hello|hey|okay|so)[,.!]?\s+)?(?i:this is|it's)\s+{_NAME}"),
    re.compile(rf"(?:^|[.,!?]\s*){_NAME}\s+here\b"),
]
ADDRESS_PREV = re.compile(
    rf"^(?:thanks|thank you|okay|ok|great|right|agreed|good point)[,!]?\s+{_NAME}", re.IGNORECASE
)
ADDRESS_NEXT = re.compile(rf"\b{_NAME},\s+(?:can|could|would|will|please|do|are|what|how)\b")

NOT_NAMES = frozenset(
    "Thanks Okay Great Sure Yes Yeah Hi Hello Hey Right Fine Good So Well Actually Also Next "
    "Let Lets Let's Sounds Someone Everyone Anyone Nobody One Quick Then That This There It "
    "We They I Agreed Monday Tuesday Wednesday Thursday Friday Saturday Sunday Postgres MySQL "
    "Going Not Just Done Here Sorry Happy Glad "
    # job titles people introduce themselves with ("I'm User Interface designer")
    "User Project Marketing Industrial Interface Designer Manager Expert Lead Head Chief "
    "Engineer Developer Product Sales Team Your Our The A An "
    # fillers and interjections
    "Um Uh Er Erm Hmm Mm Mhm Oh Ah Yep Nope Cool Alright Anyway Which What Where When Why How "
    "And But Or Wait Yes No Please".split()
)

WEIGHTS = {"self_introduction": 3.0, "direct_address": 1.0}


@dataclass
class SpeakerName:
    label: str
    name: str
    method: str
    confidence: float
    votes: float


def _accept(name: str, participants: set[str] | None) -> bool:
    if name in NOT_NAMES:
        return False
    return participants is None or name.lower() in participants


def detect_names(
    transcript: Transcript,
    participants: list[str] | None = None,
    known_names: set[str] | None = None,
) -> dict[str, SpeakerName]:
    """`known_names`: names established elsewhere (e.g. earlier meetings in the series);
    they corroborate weak signals the same way a mid-sentence mention does."""
    allowed = {p.split()[0].lower() for p in participants} if participants else None
    known = {n.lower() for n in known_names or ()}
    votes: dict[str, Counter] = defaultdict(Counter)
    methods: dict[tuple[str, str], str] = {}
    utts = transcript.utterances
    # Words written capitalized mid-sentence somewhere in the meeting. Fillers that
    # look like addressed names ("Um, can you…", "Cool, what…") only occur at
    # sentence starts, so direct address must be corroborated by this set.
    mid_caps = {
        m.group(1)
        for u in utts
        for m in re.finditer(r"(?<![.!?]\s)(?<!^)(?<=\s)([A-Z][a-z]{1,20})\b", u.text)
    }

    def addressable(name: str) -> bool:
        return _accept(name, allowed) and (
            name in mid_caps or allowed is not None or name.lower() in known
        )

    def vote(label: str, name: str, method: str) -> None:
        votes[label][name] += WEIGHTS[method]
        key = (label, name)
        if methods.get(key) != "self_introduction":
            methods[key] = method

    for i, u in enumerate(utts):
        for pattern in SELF_INTRO:
            for m in pattern.finditer(u.text):
                if _accept(m.group(1), allowed):
                    vote(u.speaker_label, m.group(1), "self_introduction")
        for pattern in WEAK_INTRO:
            for m in pattern.finditer(u.text):
                if addressable(m.group(1)):
                    vote(u.speaker_label, m.group(1), "self_introduction")
        m = ADDRESS_PREV.search(u.text)
        if m and i > 0 and addressable(m.group(1).capitalize()):
            prev = utts[i - 1]
            if prev.speaker_label != u.speaker_label:
                vote(prev.speaker_label, m.group(1).capitalize(), "direct_address")
        for m in ADDRESS_NEXT.finditer(u.text):
            if i + 1 < len(utts) and addressable(m.group(1)):
                nxt = utts[i + 1]
                if nxt.speaker_label != u.speaker_label:
                    vote(nxt.speaker_label, m.group(1), "direct_address")

    # Resolve greedily by vote strength so a name is never given to two speakers.
    candidates = sorted(
        ((v, label, name) for label, c in votes.items() for name, v in c.items()), reverse=True
    )
    out: dict[str, SpeakerName] = {}
    used: set[str] = set()
    for v, label, name in candidates:
        if label in out or name in used:
            continue
        total = sum(votes[label].values())
        method = methods[(label, name)]
        base = 0.9 if method == "self_introduction" else 0.6
        confidence = round(min(0.99, base * (v / total) + 0.05 * min(v, 3)), 3)
        out[label] = SpeakerName(label, name, method, confidence, v)
        used.add(name)
    return out


def name_speakers(
    transcript: Transcript,
    participants: list[str] | None = None,
    min_confidence: float = 0.5,
    known_names: set[str] | None = None,
) -> Transcript:
    """Return a copy with `speaker_name` filled where a name is confidently known."""
    names = {
        label: sn.name
        for label, sn in detect_names(transcript, participants, known_names).items()
        if sn.confidence >= min_confidence
    }
    utterances = [
        u.model_copy(update={"speaker_name": u.speaker_name or names.get(u.speaker_label)})
        for u in transcript.utterances
    ]
    return transcript.model_copy(update={"utterances": utterances})
