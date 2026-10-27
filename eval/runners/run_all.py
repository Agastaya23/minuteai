"""Run every experiment and write a consolidated markdown report.

    cd backend && uv run python ../eval/runners/run_all.py     # AMI (+QMSum if present)
    cd backend && uv run python ../eval/runners/run_all.py --dataset samples --replay

Sections: ablation (A/B/C), human-labelled hallucination + inter-rater kappa,
drop-reason taxonomy, action items & owners, summarization, search (hit@k/MRR),
commitment lifecycle accuracy, cost/latency per meeting-hour.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.common import DATA_DIR, RESULTS_DIR, ensure_dirs  # noqa: E402
from eval.metrics import agreement, commitments  # noqa: E402
from eval.metrics import cost as cost_metrics  # noqa: E402
from eval.metrics import grounding as grounding_metrics  # noqa: E402
from eval.metrics import search as search_metrics  # noqa: E402
from eval.runners import run_ablation  # noqa: E402

log = logging.getLogger("run_all")

SAMPLE_QUERIES = [
    ("Why did the team choose Postgres?", "demo-sprint-planning", [(2, 6)]),
    ("Who will write the migration script?", "demo-sprint-planning", [(7, 7)]),
    ("Which port will the staging database use?", "demo-sprint-planning", [(8, 8)]),
    ("What is the fraud team's latency budget?", "demo-sprint-review", [(4, 4)]),
    ("When did staging go live?", "demo-sprint-retro", [(1, 1)]),
    ("Who owns the asynchronous fraud check?", "demo-sprint-review", [(7, 7)]),
    ("What happened to the Thursday demos?", "demo-sprint-retro", [(5, 5)]),
    ("Who will write the rollback plan?", "demo-sprint-review", [(11, 11)]),
]


def fmt(v, digits: int = 3) -> str:
    if isinstance(v, float):
        return "—" if math.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


# -- search -------------------------------------------------------------------

def eval_search(dataset: str, n: int | None) -> dict | None:
    from minuteai.models.meeting import Meeting
    from minuteai.search.hybrid import HybridRetriever
    from minuteai.search.index import SearchIndex

    index = SearchIndex()
    queries: list[tuple[str, str, list[tuple[int, int]]]] = []
    if dataset == "samples":
        from minuteai.samples import SAMPLE_MEETINGS, sample_transcript

        for mid, spec in SAMPLE_MEETINGS.items():
            index.add_meeting(Meeting(meeting_id=mid, title=spec["title"]),
                              sample_transcript(mid), None)
        queries = SAMPLE_QUERIES
    else:
        from eval.datasets import qmsum

        meetings = qmsum.load(n=n)
        if not meetings:
            return None
        for m in meetings:
            index.add_meeting(Meeting(meeting_id=m.meeting_id, title=m.meeting_id),
                              m.transcript, None)
            queries += [(q.query, q.meeting_id, q.relevant_spans) for q in m.queries
                        if q.relevant_spans]
    retriever = HybridRetriever(index)
    results = []
    for query, mid, spans in queries:
        hits = retriever.retrieve(query, k=10)
        retrieved = [(h.entry.start_utt, h.entry.end_utt) if h.entry.meeting_id == mid
                     else (-10, -10) for h in hits]
        results.append((retrieved, spans))
    return {"embedder": index.embedder.name, **search_metrics.evaluate(results)}


# -- commitments ----------------------------------------------------------------

def eval_commitments(dataset: str) -> dict | None:
    if dataset != "samples":
        path = DATA_DIR / "commitment_labels.csv"
        return None if not path.exists() else {"note": f"labels at {path}; run the series "
                                                       "through the graph and compare"}
    from datetime import timedelta

    from eval.datasets.samples import GOLD_COMMITMENT_STATES
    from minuteai.extract.extractor import Extractor
    from minuteai.extract.llm_client import LLMClient
    from minuteai.extract.speakers import name_speakers
    from minuteai.graph.entity_resolution import MeetingBundle, PeopleIndex, resolve_people
    from minuteai.graph.linker import link_commitments
    from minuteai.graph.similarity import Similarity
    from minuteai.ground.filter import GroundingFilter
    from minuteai.models.meeting import Meeting
    from minuteai.samples import SAMPLE_MEETINGS, ReplayProvider, sample_transcript
    from minuteai.search.embedder import get_embedder

    start = datetime(2026, 10, 13, 10, tzinfo=UTC)
    bundles = []
    for i, (mid, spec) in enumerate(SAMPLE_MEETINGS.items()):
        t = name_speakers(sample_transcript(mid))
        m = Meeting(meeting_id=mid, title=spec["title"], occurred_at=start + timedelta(days=7 * i))
        summary = Extractor(llm=LLMClient(ReplayProvider(mid), log_path=RESULTS_DIR / "x.jsonl")
                            ).extract_meeting(t, occurred_at=m.occurred_at)
        kept, _ = GroundingFilter(t, log_path=RESULTS_DIR / "x.jsonl").apply(summary)
        bundles.append(MeetingBundle(m, t, kept))
    (RESULTS_DIR / "x.jsonl").unlink(missing_ok=True)
    sim = Similarity(get_embedder())
    found = link_commitments("eval", bundles, PeopleIndex(resolve_people(bundles)), sim)
    predicted, gold = {}, {}
    for (origin, needle), state in GOLD_COMMITMENT_STATES.items():
        c = next((c for c in found if c.origin_meeting_id == origin
                  and needle in c.description.lower()), None)
        if c is not None:
            predicted[c.commitment_id], gold[c.commitment_id] = c.state, state
        else:
            gold[f"missing:{needle}"] = state
            predicted[f"missing:{needle}"] = "COMMITTED"
    return commitments.evaluate(predicted, gold)


# -- report -------------------------------------------------------------------------

def write_report(sections: dict, args) -> Path:
    lines = ["# MinuteAI evaluation report",
             "",
             f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · dataset **{args.dataset}** · "
             f"transcript **{args.transcript}** · "
             f"LLM **{sections['ablation']['model']}** · chunk size "
             f"{sections['ablation']['chunk_tokens']} tokens",
             ""]
    if args.replay:
        lines += ["> ⚠️ Offline smoke test on recorded sample responses. These numbers validate "
                  "the harness, they are **not** experimental results.", ""]

    abl = sections["ablation"]
    lines += [f"Meetings: {', '.join(abl['meetings'])}", ""]
    if abl.get("skipped"):
        lines += ["Skipped: " + "; ".join(f"{k} ({v[:80]})" for k, v in abl["skipped"].items()),
                  ""]
    lines += ["## 1. Ablation: does grounding reduce hallucination?", "",
              "A = single prompt, no evidence · B = chunked extraction with evidence, no filter · "
              "C = B + grounding filter in each strictness mode.", "",
              md_table(abl["table"], ["config", "artifacts", "retention", "validator_pass_rate",
                                      "unsupported_entity_rate", "llm_judge_hallucination_rate",
                                      "llm_judge_n", "hallucination_rate", "labelled_n",
                                      "rougeL", "action_f1"]), "",
              "* **hallucination_rate** comes from human labels in `labelling_sheet.csv` "
              "(unsupported + contradicted) / labelled; blank until labels are filled in.",
              "* **llm_judge_hallucination_rate** is the same measure labelled automatically by "
              "an LLM judge (`run_autojudge.py`): a preliminary proxy, not ground truth.",
              "* **unsupported_entity_rate** is the automatic proxy: the share of artifacts with a "
              "name, number or date that appears nowhere in the transcript.", "",
              "![](figures/unsupported_entity_rate.png)", "",
              "![](figures/strictness_tradeoff.png)", ""]

    lines += ["## 2. Why generations fail grounding", ""]
    for mode, reasons in abl["drop_reasons"].items():
        if reasons:
            ranked = sorted(reasons.items(), key=lambda kv: -kv[1])
            lines.append(f"* **{mode}**: " + ", ".join(f"{k} × {v}" for k, v in ranked))
    lines += ["", "![](figures/drop_reasons.png)", ""]

    agr = sections.get("agreement")
    lines += ["## 3. Inter-rater agreement", ""]
    lines += [f"Cohen's κ = **{fmt(agr['kappa'])}** ({agr['interpretation']}) over "
              f"{agr['n']} doubly-labelled statements.", ""] if agr else \
        ["Fill `rater_1` and `rater_2` on an overlapping subset of `labelling_sheet.csv`.", ""]

    lines += ["## 4. Action items and owners", "",
              md_table(abl["table"], ["config", "action_precision", "action_recall", "action_f1",
                                      "action_owner_accuracy", "action_abstention_rate",
                                      "action_wrong_assignment_rate"]), "",
              "Abstention (\"unassigned\") and wrong assignment are reported separately: an honest "
              "abstention is a different failure from a confident wrong owner.", ""]

    lines += ["## 5. Summarization (ROUGE vs gold abstracts)", "",
              md_table(abl["table"], ["config", "rouge1", "rouge2", "rougeL"]), ""]

    srch = sections.get("search")
    lines += ["## 6. Retrieval", ""]
    if srch:
        lines += [md_table([srch], ["embedder", "n", "hit@1", "hit@3", "hit@5", "hit@10", "mrr"]),
                  ""]
    else:
        lines += ["QMSum not found — clone it into `eval/data/qmsum`.", ""]

    com = sections.get("commitments")
    lines += ["## 7. Commitment lifecycle", ""]
    if com and "accuracy" in com:
        lines += [f"Accuracy **{fmt(com['accuracy'])}** over {com['n']} labelled commitments.", "",
                  "| gold \\ predicted | " + " | ".join(commitments.STATES) + " |",
                  "|---|" + "---|" * len(commitments.STATES)]
        for g in commitments.STATES:
            row = com["confusion"].get(g, {})
            if sum(row.values()):
                lines.append(f"| {g} | " + " | ".join(str(row.get(p, 0)) for p in
                                                      commitments.STATES) + " |")
        lines.append("")
    else:
        lines += ["Label `eval/data/commitment_labels.csv` (commitment_id,gold_state) to score "
                  "the lifecycle classifier.", ""]

    cost = sections["cost"]
    t = cost["totals"]
    lines += ["## 8. Cost and latency", "",
              f"{t['calls']} LLM calls · {t['input_tokens']:,} input / {t['output_tokens']:,} "
              f"output tokens · ${t['cost_usd']:.4f} · {t['latency_ms'] / 1000:.1f}s total latency "
              f"· {cost['retries']} calls needed a parse retry.", ""]
    if "per_meeting_hour" in cost:
        p = cost["per_meeting_hour"]
        lines += [f"Per meeting-hour: {p['tokens']:,.0f} tokens · ${p['cost_usd']:.4f} · "
                  f"{p['latency_s']:.1f}s of LLM latency.", ""]
    path = RESULTS_DIR / "report.md"
    path.write_text("\n".join(lines))
    return path


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", choices=["ami", "samples"], default="ami")
    p.add_argument("--meetings", type=int, default=None)
    p.add_argument("--transcript", choices=["reference", "asr"], default="reference")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--qmsum-meetings", type=int, default=None)
    p.add_argument("--model", default=None, help="override LLM_MODEL")
    p.add_argument("--chunk-tokens", type=int, default=6000)
    args = p.parse_args(argv)
    ensure_dirs()

    ablation_args = ["--dataset", args.dataset, "--transcript", args.transcript]
    if args.meetings:
        ablation_args += ["--meetings", str(args.meetings)]
    if args.replay:
        ablation_args.append("--replay")
    if args.model:
        ablation_args += ["--model", args.model]
    ablation_args += ["--chunk-tokens", str(args.chunk_tokens)]
    sections: dict = {"ablation": run_ablation.main(ablation_args)}
    judge_path = RESULTS_DIR / "autojudge.json"
    if judge_path.exists():  # written by run_autojudge.py (LLM-judge proxy, not human)
        judge = json.loads(judge_path.read_text())
        for row in sections["ablation"]["table"]:
            r = judge.get(row["config"])
            row["llm_judge_hallucination_rate"] = r["hallucination_rate"] if r else float("nan")
            row["llm_judge_n"] = r["n"] if r else 0

    labels = grounding_metrics.read_labels(RESULTS_DIR / "labelling_sheet.csv")
    both = [(r["rater_1"].strip().lower(), r["rater_2"].strip().lower()) for r in labels
            if r.get("rater_1", "").strip() and r.get("rater_2", "").strip()]
    if both:
        k = agreement.cohens_kappa([a for a, _ in both], [b for _, b in both])
        sections["agreement"] = {"kappa": k, "n": len(both),
                                 "interpretation": agreement.interpret(k)}

    sections["search"] = eval_search(args.dataset, args.qmsum_meetings)
    sections["commitments"] = eval_commitments(args.dataset)

    calls = []
    from eval.common import CACHE_DIR

    for config in ("A", "B"):
        calls += cost_metrics.load_calls(CACHE_DIR / f"llm_calls_{config}.jsonl")
    sections["cost"] = cost_metrics.summarize(calls, sections["ablation"]["hours"])

    (RESULTS_DIR / "all_results.json").write_text(json.dumps(sections, indent=2, default=str))
    print(f"report → {write_report(sections, args)}")


if __name__ == "__main__":
    main()
