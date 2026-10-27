"""Accountability graph over an AMI series (e.g. ES2002a–d), from cached extractions.

    cd backend && uv run python ../eval/runners/run_graph.py --series ES2002

Uses config C-BALANCED (B's cached extraction through the BALANCED filter), so no
new LLM calls are made. Writes:
  eval/results/graph_<series>.json         full graph (commitments, transitions, reversals)
  eval/results/commitment_labels_<series>.csv  template to fill `gold_state` by hand
and, if eval/data/commitment_labels.csv exists, scores lifecycle accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.common import DATA_DIR, RESULTS_DIR, cache_key, ensure_dirs  # noqa: E402
from eval.datasets import ami  # noqa: E402
from eval.metrics import commitments as commitment_metrics  # noqa: E402
from minuteai.config import get_settings  # noqa: E402
from minuteai.extract.speakers import name_speakers  # noqa: E402
from minuteai.graph.decisions import detect_reversals  # noqa: E402
from minuteai.graph.entity_resolution import (  # noqa: E402
    MeetingBundle,
    PeopleIndex,
    resolve_people,
)
from minuteai.graph.linker import link_commitments  # noqa: E402
from minuteai.graph.similarity import Similarity  # noqa: E402
from minuteai.ground.filter import GroundingFilter  # noqa: E402
from minuteai.models.artifacts import MeetingSummary  # noqa: E402
from minuteai.models.meeting import Meeting  # noqa: E402
from minuteai.search.embedder import get_embedder  # noqa: E402


def cached_b(meeting_id: str, transcript, model: str, chunk_tokens: int) -> MeetingSummary | None:
    key = cache_key(meeting_id, "B", model, transcript.model_version,
                    len(transcript.utterances), chunk_tokens)
    path = Path(__file__).resolve().parents[1] / "cache" / "extract" / "B" / f"{key}.json"
    return MeetingSummary.model_validate_json(path.read_text()) if path.exists() else None


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--series", default="ES2002")
    p.add_argument("--model", default=None, help="model used by run_ablation (cache key)")
    p.add_argument("--chunk-tokens", type=int, default=6000)
    args = p.parse_args(argv)
    ensure_dirs()

    # AMI sessions a–d happened on separate days; one week apart is a neutral assumption.
    start = datetime(2005, 1, 3, 10, tzinfo=UTC)
    bundles, missing = [], []
    for i, suffix in enumerate("abcd"):
        mid = f"{args.series}{suffix}"
        m = ami.load_meeting(mid)
        if m.transcript is None:
            continue
        b = cached_b(mid, m.transcript, args.model or get_settings().llm_model,
                     args.chunk_tokens)
        if b is None:
            missing.append(mid)
            continue
        t = name_speakers(m.transcript)
        kept, _ = GroundingFilter(t, mode="BALANCED",
                                  log_path=RESULTS_DIR / "graph_dropped.jsonl").apply(b)
        meeting = Meeting(meeting_id=mid, title=mid, series_id=args.series,
                          occurred_at=start + timedelta(days=7 * i))
        bundles.append(MeetingBundle(meeting, t, kept))
    (RESULTS_DIR / "graph_dropped.jsonl").unlink(missing_ok=True)
    if missing:
        print(f"no cached extraction for {missing}: run run_ablation.py on them first")
    if not bundles:
        sys.exit("nothing to build")

    sim = Similarity(get_embedder())
    people = resolve_people(bundles)
    commitments = link_commitments(args.series, bundles, PeopleIndex(people), sim)
    reversals = detect_reversals(args.series, bundles, sim)

    out = {
        "series": args.series,
        "meetings": [b.meeting_id for b in bundles],
        "people": [p.model_dump() for p in people.values()],
        "commitments": [c.model_dump(mode="json") for c in commitments],
        "reversals": [r.model_dump(mode="json") for r in reversals],
        "state_counts": dict(Counter(c.state for c in commitments)),
        "linked_across_meetings": sum(len({m.meeting_id for m in c.mentions}) > 1
                                      for c in commitments),
    }
    (RESULTS_DIR / f"graph_{args.series}.json").write_text(json.dumps(out, indent=2))
    commitment_metrics.write_label_template(
        {c.commitment_id: c.state for c in commitments},
        {c.commitment_id: c.description for c in commitments},
        RESULTS_DIR / f"commitment_labels_{args.series}.csv",
    )
    labels = DATA_DIR / "commitment_labels.csv"
    if labels.exists():
        out["accuracy"] = commitment_metrics.evaluate(
            {c.commitment_id: c.state for c in commitments},
            commitment_metrics.load_labels(labels))
    print(json.dumps({k: out[k] for k in ("series", "meetings", "state_counts",
                                          "linked_across_meetings")}, indent=2))
    print(f"people: {[(p.display_name or p.person_id, p.method) for p in people.values()]}")
    print(f"reversals: {len(reversals)}")
    for c in commitments:
        print(f"  [{c.state:<16}] {c.description[:70]}  ({c.owner_name or 'unassigned'})")
    return out


if __name__ == "__main__":
    main()
