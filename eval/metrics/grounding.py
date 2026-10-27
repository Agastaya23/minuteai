"""The headline experiment: how often do generated statements lack support?

1. `write_labelling_sheet` samples N generated statements stratified across
   (config, meeting, artifact type) and writes eval/results/labelling_sheet.csv
   with the cited utterances alongside, for humans to label
   supported / unsupported / contradicted (rater_1, rater_2 columns).
2. `hallucination_rates` reads the labelled sheet back:
   hallucination rate = (unsupported + contradicted) / labelled, per config.
3. `retention` and `drop_reasons` summarize the filter's behaviour.
4. `transcript_entity_support` is an *automatic* proxy usable on every config
   (including A, which cites nothing): the fraction of artifacts whose names,
   numbers and dates all occur somewhere in the transcript.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from minuteai.ground.validator import GroundingValidator, artifact_text, normalize
from minuteai.models.artifacts import MeetingSummary
from minuteai.models.transcript import Transcript

LABELS = ("supported", "unsupported", "contradicted")
SHEET_FIELDS = ["item_id", "config", "meeting_id", "artifact_type", "text", "cited_evidence",
                "rater_1", "rater_2", "notes"]


@dataclass
class Statement:
    config: str
    meeting_id: str
    artifact_type: str
    artifact_id: str
    text: str
    artifact: object


def statements(config: str, summary: MeetingSummary) -> list[Statement]:
    out = [Statement(config, summary.meeting_id, "abstract", f"s{i}", s.text, s)
           for i, s in enumerate(summary.abstract)]
    out += [Statement(config, summary.meeting_id, "decision", d.decision_id, artifact_text(d), d)
            for d in summary.decisions]
    out += [Statement(config, summary.meeting_id, "action_item", a.item_id, _action_text(a), a)
            for a in summary.action_items]
    return out


def _action_text(a) -> str:
    """Action items are labelled as full claims, including who and when."""
    details = [f"owner: {a.owner}" if a.owner else "owner: unassigned"]
    if a.due_raw:
        details.append(f"due: {a.due_raw}")
    return f"{a.description} ({'; '.join(details)})"


def stratified_sample(items: list[Statement], n: int, seed: int = 13) -> list[Statement]:
    """Round-robin across (config, meeting, type) strata so no stratum dominates."""
    rng = random.Random(seed)
    strata: dict[tuple, list[Statement]] = defaultdict(list)
    for s in items:
        strata[(s.config, s.meeting_id, s.artifact_type)].append(s)
    for bucket in strata.values():
        rng.shuffle(bucket)
    keys = sorted(strata)
    picked: list[Statement] = []
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    rng.shuffle(picked)  # raters must not see items grouped by config
    return picked


def _evidence_text(s: Statement, transcripts: dict[str, Transcript]) -> str:
    t = transcripts.get(s.meeting_id)
    parts = []
    for e in getattr(s.artifact, "evidence", []) or []:
        u = t.get(e.utt_id) if t else None
        parts.append(f"[{e.utt_id}] {u.speaker if u else '?'}: {u.text if u else '(missing)'}")
    return "\n".join(parts) or "(no evidence cited — check the full transcript)"


def write_labelling_sheet(sample: list[Statement], transcripts: dict[str, Transcript],
                          path: Path) -> Path:
    from minuteai.extract.speakers import name_speakers

    transcripts = {k: name_speakers(t) for k, t in transcripts.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        w.writeheader()
        for i, s in enumerate(sample):
            w.writerow({"item_id": f"g{i:04d}", "config": s.config, "meeting_id": s.meeting_id,
                        "artifact_type": s.artifact_type, "text": s.text,
                        "cited_evidence": _evidence_text(s, transcripts),
                        "rater_1": "", "rater_2": "", "notes": ""})
    return path


def read_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def hallucination_rates(rows: list[dict], rater: str = "rater_1") -> dict[str, dict]:
    by_config: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        label = (r.get(rater) or "").strip().lower()
        if label in LABELS:
            by_config[r["config"]][label] += 1
    out = {}
    for config, c in sorted(by_config.items()):
        n = sum(c.values())
        out[config] = {"n": n, **{k: c[k] for k in LABELS},
                       "hallucination_rate": (c["unsupported"] + c["contradicted"]) / n
                       if n else float("nan")}
    return out


def derived_filter_rates(
    rows: list[dict], label_col: str, kept: dict[str, set[tuple[str, str]]],
    source_config: str = "B",
) -> dict[str, dict]:
    """Hallucination rate of each filter mode measured on the *same* labelled items.

    Every artifact the filter keeps (config C-*) is an artifact of config B, so C's rate
    is the rate among labelled B items that the mode keeps. This compares B and C on
    identical statements instead of on two independent samples.
    `kept[config]` = {(meeting_id, statement text)} that config retained.
    """
    labelled = [r for r in rows if r["config"] == source_config
                and (r.get(label_col) or "").strip().lower() in LABELS]
    out = {}
    for config, keep in kept.items():
        c = Counter(r[label_col].strip().lower() for r in labelled
                    if (r["meeting_id"], r["text"]) in keep)
        n = sum(c.values())
        out[config] = {"n": n, **{k: c[k] for k in LABELS},
                       "hallucination_rate": (c["unsupported"] + c["contradicted"]) / n
                       if n else float("nan")}
    return out


def retention(n_total: int, n_kept: int) -> float:
    return n_kept / n_total if n_total else float("nan")


def drop_reasons(dropped_log: Path, meeting_ids: set[str] | None = None) -> Counter:
    c: Counter = Counter()
    if not dropped_log.exists():
        return c
    for line in dropped_log.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if meeting_ids and d.get("meeting_id") not in meeting_ids:
            continue
        for f in d.get("failures", []):
            c[f["reason"]] += 1
    return c


def transcript_entity_support(summary: MeetingSummary, transcript: Transcript) -> dict[str, float]:
    """Fraction of artifacts whose entities all appear *anywhere* in the transcript."""
    validator = GroundingValidator(transcript, use_spacy=False)
    corpus = normalize(" ".join(
        f"{u.text} {u.speaker_name or ''}" for u in transcript.utterances))
    items = statements("x", summary)
    if not items:
        return {"n": 0, "entity_supported_rate": float("nan")}
    ok = 0
    for s in items:
        ents = validator.extract_entities(s.text)
        ok += all(validator._supported(e, corpus, []) for e in ents)
    return {"n": len(items), "entity_supported_rate": ok / len(items)}


def validator_pass_rate(summary: MeetingSummary, transcript: Transcript) -> dict[str, float]:
    v = GroundingValidator(transcript, use_spacy=False)
    items = statements("x", summary)
    if not items:
        return {"n": 0, "pass_rate": float("nan"), "mean_confidence": float("nan")}
    results = [v.validate(s.artifact) for s in items]
    return {"n": len(items), "pass_rate": sum(r.passed for r in results) / len(items),
            "mean_confidence": sum(r.confidence for r in results) / len(items)}
