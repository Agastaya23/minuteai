"""Automatic LLM-judge labels for the labelling sheet — a *proxy*, not human ground truth.

    cd backend && uv run python ../eval/runners/run_autojudge.py

Reads eval/results/labelling_sheet.csv and writes labelling_sheet.autojudge.csv with an
`llm_judge` column (supported / unsupported / contradicted) and a reason, plus
autojudge.json with per-config rates. The rater_1 / rater_2 columns are never touched:
the headline hallucination rate must come from people. The judge sees the same
material a human rater would: the cited utterances with ±3 utterances of context, or
the whole transcript for statements that cite nothing (config A).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.common import RESULTS_DIR  # noqa: E402
from eval.datasets import ami, samples  # noqa: E402
from minuteai.extract.chunker import render_utterances  # noqa: E402
from minuteai.extract.llm_client import (  # noqa: E402
    LLMClient,
    LLMError,
    LLMQuotaExhaustedError,
)
from minuteai.extract.speakers import name_speakers  # noqa: E402

CONTEXT = 3
JUDGE_PROMPT = """\
You are checking meeting minutes against the meeting transcript.

Statement from the minutes:
  "{statement}"

Transcript excerpt:
{excerpt}

Label the statement:
- "supported": everything it asserts (including names, owners, numbers, dates) is stated or
  directly implied by the excerpt.
- "unsupported": some part of it is not in the excerpt (invented or unverifiable detail).
- "contradicted": the excerpt says something incompatible with it.

Return JSON: {{"label": "supported"|"unsupported"|"contradicted", "reason": str}}
"""


class JudgeLabel(BaseModel):
    label: Literal["supported", "unsupported", "contradicted"]
    reason: str


def transcripts_for(meeting_ids: set[str]):
    out = {}
    sample_ids = {m.meeting_id: m for m in samples.load()}
    for mid in meeting_ids:
        m = sample_ids.get(mid) or ami.load_meeting(mid)
        if m.transcript is not None:
            out[mid] = name_speakers(m.transcript)
    return out


def excerpt(row: dict, transcript) -> str:
    import re

    cited = [int(x) for x in re.findall(r"^\[(\d+)\]", row["cited_evidence"], re.MULTILINE)]
    if not cited:
        return render_utterances(transcript.utterances)
    keep: set[int] = set()
    for u in cited:
        keep |= set(range(u - CONTEXT, u + CONTEXT + 1))
    return render_utterances([u for u in transcript.utterances if u.utt_id in keep])


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sheet", type=Path, default=RESULTS_DIR / "labelling_sheet.csv")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default=None,
                   help="judge model; ideally different from the extraction model")
    args = p.parse_args(argv)
    rows = list(csv.DictReader(args.sheet.open()))[: args.limit]
    out_path = args.sheet.with_suffix(".autojudge.csv")
    done = {}
    if out_path.exists():  # resumable: free-tier rate limits make long runs fragile
        done = {r["item_id"]: r for r in csv.DictReader(out_path.open()) if r.get("llm_judge")}

    transcripts = transcripts_for({r["meeting_id"] for r in rows})
    llm = LLMClient(log_path=RESULTS_DIR.parent / "cache" / "llm_calls_judge.jsonl",
                    model=args.model)
    fields = [*rows[0].keys(), "llm_judge", "llm_judge_reason"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            if r["item_id"] in done:
                w.writerow(done[r["item_id"]])
                continue
            t = transcripts.get(r["meeting_id"])
            label, reason = "", "transcript unavailable"
            if t is not None:
                try:
                    j = llm.complete_structured(
                        JUDGE_PROMPT.format(statement=r["text"], excerpt=excerpt(r, t)),
                        JudgeLabel, purpose="judge", meeting_id=r["meeting_id"])
                    label, reason = j.label, j.reason
                except LLMQuotaExhaustedError as e:
                    print(f"stopping: {e} (re-run later; finished rows are kept)")
                    w.writerow({**r, "llm_judge": "", "llm_judge_reason": "not judged yet"})
                    for rest in rows[i + 1:]:
                        w.writerow(done.get(rest["item_id"], {**rest, "llm_judge": "",
                                                               "llm_judge_reason": ""}))
                    break
                except LLMError as e:
                    reason = f"judge failed: {e}"[:300]
            w.writerow({**r, "llm_judge": label, "llm_judge_reason": reason})
            f.flush()
            print(f"{i + 1}/{len(rows)} {r['config']:<11} {label or '—':<13} {r['text'][:60]}")

    labelled = [r for r in csv.DictReader(out_path.open()) if r.get("llm_judge")]
    by_config: dict[str, Counter] = defaultdict(Counter)
    for r in labelled:
        by_config[r["config"]][r["llm_judge"]] += 1
    rates = {
        c: {"n": sum(v.values()), **v,
            "hallucination_rate": (v["unsupported"] + v["contradicted"]) / sum(v.values())}
        for c, v in sorted(by_config.items())
    }
    (RESULTS_DIR / "autojudge.json").write_text(json.dumps(rates, indent=2))
    print(json.dumps(rates, indent=2))
    return rates


if __name__ == "__main__":
    main()
