"""Cross-meeting identity resolution.

Speaker labels are per-meeting ("SPEAKER_01" is a different person each time).
They are resolved to stable identities using, in order of strength:
  * self-introductions in the transcript        ("I'm Priya", "Rahul here")
  * names used in direct address                 ("Thanks Rahul", "Ananya, can you…")
  * the participant list supplied at upload      (constrains and confirms names)
  * voice-embedding similarity (pyannote)        (last resort for unnamed speakers)
Every resolution carries the method and a confidence; anything below the
threshold stays unresolved rather than being guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minuteai.extract.speakers import detect_names
from minuteai.models.artifacts import MeetingSummary
from minuteai.models.graph import PersonIdentity
from minuteai.models.meeting import Meeting
from minuteai.models.transcript import Transcript
from minuteai.storage.schema import normalize_name

MIN_CONFIDENCE = 0.5
VOICE_THRESHOLD = 0.75
METHOD_RANK = {"self_introduction": 4, "participant_list": 3, "direct_address": 2,
               "voice_embedding": 1, "owner_name": 1, "unresolved": 0}


@dataclass
class MeetingBundle:
    meeting: Meeting
    transcript: Transcript
    summary: MeetingSummary | None

    @property
    def meeting_id(self) -> str:
        return self.meeting.meeting_id


class PeopleIndex:
    """person_id → identity, plus (meeting_id, speaker_label) → person_id."""

    def __init__(self, people: dict[str, PersonIdentity]) -> None:
        self.people = people
        self.by_label: dict[tuple[str, str], str] = {
            (mid, label): pid
            for pid, p in people.items()
            for mid, label in p.speaker_labels.items()
        }
        self.by_name: dict[str, str] = {}
        for pid, p in people.items():
            for alias in [p.display_name, *p.aliases]:
                if alias:
                    self.by_name.setdefault(normalize_name(alias), pid)

    def for_label(self, meeting_id: str, label: str | None) -> str | None:
        return self.by_label.get((meeting_id, label)) if label else None

    def for_name(self, name: str | None) -> str | None:
        if not name:
            return None
        key = normalize_name(name)
        return self.by_name.get(key) or self.by_name.get(key.split("_")[0])

    def name(self, person_id: str | None) -> str | None:
        p = self.people.get(person_id) if person_id else None
        return p.display_name if p else None


def _unresolved_id(meeting_id: str, label: str) -> str:
    return f"unresolved-{meeting_id}-{label}".lower()


def resolve_people(bundles: list[MeetingBundle]) -> dict[str, PersonIdentity]:
    people: dict[str, PersonIdentity] = {}
    scores: dict[str, list[float]] = {}
    unnamed: list[tuple[MeetingBundle, str]] = []

    for b in bundles:
        participants = b.meeting.participants or None
        allowed = {normalize_name(p) for p in participants} if participants else None
        found = detect_names(b.transcript, participants)
        for label in b.transcript.speakers():
            sn = found.get(label)
            if sn is None or sn.confidence < MIN_CONFIDENCE:
                unnamed.append((b, label))
                continue
            pid = normalize_name(sn.name)
            method = sn.method
            conf = sn.confidence
            if allowed and pid in allowed:
                # A name the organiser also listed is corroborated by two sources.
                conf = min(0.99, conf + 0.1)
                if METHOD_RANK["participant_list"] > METHOD_RANK[method]:
                    method = "participant_list"
            p = people.setdefault(pid, PersonIdentity(person_id=pid, display_name=sn.name))
            p.speaker_labels[b.meeting_id] = label
            if METHOD_RANK[method] > METHOD_RANK[p.method]:
                p.method = method
            scores.setdefault(pid, []).append(conf)

    for pid, s in scores.items():
        people[pid].confidence = round(float(np.mean(s)), 3)

    # Voice-embedding fallback: match unnamed speakers to known voices.
    voices: dict[str, list[np.ndarray]] = {}
    for b in bundles:
        for label, vec in b.transcript.speaker_embeddings.items():
            pid = next((p for p, ident in people.items()
                        if ident.speaker_labels.get(b.meeting_id) == label), None)
            if pid:
                voices.setdefault(pid, []).append(np.asarray(vec, dtype=np.float32))
    centroids = {pid: np.mean(v, axis=0) / (np.linalg.norm(np.mean(v, axis=0)) or 1)
                 for pid, v in voices.items()}

    for b, label in unnamed:
        vec = b.transcript.speaker_embeddings.get(label)
        best, best_sim = None, 0.0
        if vec is not None and centroids:
            v = np.asarray(vec, dtype=np.float32)
            v = v / (np.linalg.norm(v) or 1)
            taken = {p.speaker_labels.get(b.meeting_id) for p in people.values()}
            for pid, c in centroids.items():
                sim = float(np.dot(v, c))
                if sim > best_sim and b.meeting_id not in people[pid].speaker_labels \
                        and label not in taken:
                    best, best_sim = pid, sim
        if best is not None and best_sim >= VOICE_THRESHOLD:
            people[best].speaker_labels[b.meeting_id] = label
            if people[best].method == "unresolved":
                people[best].method = "voice_embedding"
            continue
        uid = _unresolved_id(b.meeting_id, label)
        people[uid] = PersonIdentity(person_id=uid, display_name=None, aliases=[label],
                                     speaker_labels={b.meeting_id: label},
                                     method="unresolved", confidence=0.0)
    return people
