"""Central experiment: does evidence-carrying extraction + grounding reduce hallucination?

    cd backend && uv run python ../eval/runners/run_ablation.py --meetings 5
    cd backend && uv run python ../eval/runners/run_ablation.py --dataset samples --replay

Configurations, run over the same meetings:
  A            raw transcript in one prompt, no chunking, no evidence requested
  B            chunked structured extraction with evidence, grounding filter OFF
  C-STRICT     B + grounding filter (any failed check drops)
  C-BALANCED   B + grounding filter (default mode)
  C-PERMISSIVE B + grounding filter (flags only)

C reuses B's extraction, so B→C isolates the filter. Transcripts and
extractions are cached in eval/cache/, so re-runs cost nothing.

Outputs: eval/results/ablation.csv, eval/results/ablation.json,
eval/results/labelling_sheet.csv (if absent), eval/results/figures/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.common import (  # noqa: E402
    CACHE_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    EvalMeeting,
    cache_key,
    cached_model,
    ensure_dirs,
)
from eval.metrics import action_items as action_metrics  # noqa: E402
from eval.metrics import cost as cost_metrics  # noqa: E402
from eval.metrics import grounding as grounding_metrics  # noqa: E402
from eval.metrics import summarization  # noqa: E402
from minuteai.config import get_settings  # noqa: E402
from minuteai.extract.extractor import Extractor  # noqa: E402
from minuteai.extract.llm_client import (  # noqa: E402
    LLMClient,
    LLMError,
    LLMQuotaExhaustedError,
)
from minuteai.ground.filter import GroundingFilter, Strictness  # noqa: E402
from minuteai.models.artifacts import MeetingSummary  # noqa: E402
from minuteai.models.transcript import Transcript  # noqa: E402

log = logging.getLogger("ablation")
MODES = [Strictness.STRICT, Strictness.BALANCED, Strictness.PERMISSIVE]
CONFIGS = ["A", "B", *[f"C-{m}" for m in MODES]]


def load_meetings(dataset: str, n: int | None) -> list[EvalMeeting]:
    if dataset == "samples":
        from eval.datasets import samples

        return samples.load(n)
    if dataset == "ami":
        from eval.datasets import ami

        meetings = ami.load(n)
        if not meetings:
            sys.exit("No AMI data found. Run: uv run python ../eval/datasets/ami.py --download")
        return meetings
    sys.exit(f"unknown dataset {dataset!r}")


def get_transcript(m: EvalMeeting, source: str) -> Transcript:
    if source == "reference":
        if m.transcript is None:
            sys.exit(f"{m.meeting_id}: no reference transcript")
        return m.transcript
    if m.audio_path is None:
        sys.exit(f"{m.meeting_id}: --transcript asr needs audio")

    def run() -> Transcript:
        import tempfile

        from minuteai.transcribe.audio import to_wav16k
        from minuteai.transcribe.base import get_backend

        with tempfile.TemporaryDirectory() as tmp:
            wav = to_wav16k(m.audio_path, Path(tmp) / "a.wav")
            return get_backend().transcribe(wav, m.meeting_id)

    key = cache_key(m.meeting_id, "asr", get_settings().whisper_model)
    return cached_model("transcripts", key, Transcript, run)


def make_llm(meeting_id: str, config: str, replay: bool, model: str | None) -> LLMClient:
    log_path = CACHE_DIR / f"llm_calls_{config}.jsonl"
    if replay:
        from minuteai.samples import ReplayProvider

        return LLMClient(ReplayProvider(meeting_id), log_path=log_path)
    return LLMClient(log_path=log_path, model=model)


def extract(m: EvalMeeting, t: Transcript, config: str, args) -> MeetingSummary:
    model = "replay" if args.replay else (args.model or get_settings().llm_model)
    key = cache_key(m.meeting_id, config, model, t.model_version, len(t.utterances),
                    args.chunk_tokens)

    def run() -> MeetingSummary:
        ex = Extractor(llm=make_llm(m.meeting_id, config, args.replay, args.model),
                       max_tokens=args.chunk_tokens)
        if config == "A":
            return ex.extract_meeting_baseline(t)
        return ex.extract_meeting(t)

    return cached_model(f"extract/{config}", key, MeetingSummary, run, args.refresh)


def count(s: MeetingSummary) -> int:
    return len(s.abstract) + len(s.decisions) + len(s.action_items)


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def run(args) -> dict:
    ensure_dirs()
    meetings = load_meetings(args.dataset, args.meetings)
    dropped_logs = {m: RESULTS_DIR / f"dropped_{m}.jsonl" for m in MODES}
    for p in dropped_logs.values():
        p.unlink(missing_ok=True)

    per_config: dict[str, list[dict]] = {c: [] for c in CONFIGS}
    summaries: dict[str, dict[str, MeetingSummary]] = {c: {} for c in CONFIGS}
    transcripts: dict[str, Transcript] = {}
    skipped: dict[str, str] = {}
    for m in meetings:
        t = get_transcript(m, args.transcript)
        log.info("%s: %d utterances", m.meeting_id, len(t.utterances))
        try:
            a = extract(m, t, "A", args)
            b = extract(m, t, "B", args)
        except LLMError as e:  # e.g. daily quota: keep the meetings that did finish
            log.error("%s skipped: %s", m.meeting_id, e)
            skipped[m.meeting_id] = str(e)
            if isinstance(e, LLMQuotaExhaustedError):
                break
            continue
        transcripts[m.meeting_id] = t
        outputs = {"A": (a, count(a)), "B": (b, count(b))}
        for mode in MODES:
            kept, _ = GroundingFilter(t, mode=mode, log_path=dropped_logs[mode]).apply(b)
            outputs[f"C-{mode}"] = (kept, count(b))
        for config, (summary, total) in outputs.items():
            summaries[config][m.meeting_id] = summary
            row = {
                "meeting_id": m.meeting_id,
                "artifacts": count(summary),
                "retention": grounding_metrics.retention(total, count(summary)),
                **{f"entity_{k}": v for k, v in
                   grounding_metrics.transcript_entity_support(summary, t).items()},
                **{f"validator_{k}": v for k, v in
                   grounding_metrics.validator_pass_rate(summary, t).items()},
            }
            if m.gold_abstract:
                row.update(summarization.rouge(
                    "\n".join(s.text for s in summary.abstract), "\n".join(m.gold_abstract)))
            if m.gold_actions:
                row.update({f"action_{k}": v for k, v in action_metrics.evaluate(
                    summary.action_items, m.gold_actions, m.gold_action_owners or None).items()})
            per_config[config].append(row)

    # aggregate
    table = []
    numeric = sorted({k for rows in per_config.values() for r in rows for k, v in r.items()
                      if isinstance(v, (int, float)) and k not in ("meeting_id",)})
    for config, rows in per_config.items():
        agg = {"config": config, "meetings": len(rows)}
        for k in numeric:
            vals = [r[k] for r in rows if k in r]
            agg[k] = sum(vals) if k == "artifacts" else mean(vals)
        agg["unsupported_entity_rate"] = 1 - agg.get("entity_entity_supported_rate", float("nan"))
        table.append(agg)

    # cost per config (only calls actually made; cached runs reuse earlier logs)
    hours = {m: transcripts[m].duration_sec / 3600 for m in transcripts}
    for row in table:
        model = "replay" if args.replay else (args.model or get_settings().llm_model)
        source = "B" if row["config"].startswith("C-") else row["config"]
        calls = [c for c in cost_metrics.load_calls(CACHE_DIR / f"llm_calls_{source}.jsonl")
                 if c.get("meeting_id") in transcripts
                 and (args.replay or c.get("model") == model)]
        s = cost_metrics.summarize(calls, hours)
        row["llm_calls"] = s["totals"]["calls"]
        row["tokens"] = s["totals"]["input_tokens"] + s["totals"]["output_tokens"]
        row["cost_usd"] = s["totals"]["cost_usd"]

    # Labels: human (rater_1 in labelling_sheet.csv) and the LLM-judge proxy
    # (labelling_sheet.autojudge.csv). A and B are scored directly; each C mode on the
    # labelled B items it keeps (same statements, see derived_filter_rates).
    sheet = RESULTS_DIR / "labelling_sheet.csv"
    labels = grounding_metrics.read_labels(sheet)
    kept = {c: {(mid, st.text) for mid, s in summaries[c].items()
                for st in grounding_metrics.statements(c, s)}
            for c in CONFIGS if c.startswith("C-")}
    judged = grounding_metrics.read_labels(sheet.with_suffix(".autojudge.csv"))
    for prefix, rows, col in (("", labels, "rater_1"), ("llm_judge_", judged, "llm_judge")):
        direct = grounding_metrics.hallucination_rates(rows, col) if rows else {}
        derived = grounding_metrics.derived_filter_rates(rows, col, kept) if rows else {}
        for row in table:
            r = derived.get(row["config"]) or direct.get(row["config"])
            row[f"{prefix}hallucination_rate"] = r["hallucination_rate"] if r else float("nan")
            row[f"{prefix}{'labelled_n' if not prefix else 'n'}"] = r["n"] if r else 0
    if not any((r.get("rater_1") or "").strip() for r in labels):
        items = []
        for config in ("A", "B"):  # C ⊆ B: C is scored on the labelled B items
            for s in summaries[config].values():
                items += grounding_metrics.statements(config, s)
        if not sheet.exists():  # never replace a sheet someone may be labelling
            sample = grounding_metrics.stratified_sample(items, args.label_n)
            grounding_metrics.write_labelling_sheet(sample, transcripts, sheet)
            log.info("wrote %d items to label → %s", len(sample), sheet)

    reasons = {str(m): dict(grounding_metrics.drop_reasons(dropped_logs[m])) for m in MODES}
    write_outputs(table, reasons, args)
    return {"table": table, "drop_reasons": reasons, "meetings": list(transcripts),
            "skipped": skipped, "hours": hours, "dataset": args.dataset,
            "transcript": args.transcript, "model": "replay" if args.replay else
            (args.model or get_settings().llm_model), "chunk_tokens": args.chunk_tokens}


def fmt(v) -> str:
    if isinstance(v, float):
        return "—" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def write_outputs(table: list[dict], reasons: dict, args) -> None:
    cols = ["config", "meetings", "artifacts", "retention", "validator_pass_rate",
            "validator_mean_confidence", "unsupported_entity_rate", "hallucination_rate",
            "labelled_n", "rouge1", "rouge2", "rougeL", "action_precision", "action_recall",
            "action_f1", "action_owner_accuracy", "action_abstention_rate",
            "action_wrong_assignment_rate", "llm_calls", "tokens", "cost_usd"]
    cols = [c for c in cols if any(c in r for r in table)]
    with (RESULTS_DIR / "ablation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in table:
            w.writerow({c: fmt(r.get(c, float("nan"))) for c in cols})
    (RESULTS_DIR / "ablation.json").write_text(json.dumps(
        {"table": table, "drop_reasons": reasons, "args": vars(args)}, indent=2, default=str))
    try:
        plot(table, reasons)
    except ImportError:
        log.warning("matplotlib not installed (uv sync --extra eval); skipping figures")


def plot(table: list[dict], reasons: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = [r["config"] for r in table]
    colors = ["#8a8f98", "#5b7db1", "#2f6f4f", "#3f8f5f", "#7fbf8f"]

    def bars(metric: str, title: str, fname: str) -> None:
        vals = [r.get(metric, float("nan")) for r in table]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.bar(configs, [0 if v != v else v for v in vals], color=colors[: len(configs)])
        for i, v in enumerate(vals):
            ax.text(i, (0 if v != v else v) + 0.01, "n/a" if v != v else f"{v:.2f}",
                    ha="center", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / fname, dpi=160)
        plt.close(fig)

    bars("unsupported_entity_rate", "Artifacts with names/numbers absent from the transcript",
         "unsupported_entity_rate.png")
    bars("validator_pass_rate", "Grounding validator pass rate", "validator_pass_rate.png")
    bars("retention", "Retention (kept / extracted)", "retention.png")
    if any(r.get("hallucination_rate", float("nan")) == r.get("hallucination_rate")
           for r in table):
        bars("hallucination_rate", "Human-labelled hallucination rate", "hallucination_rate.png")
    if any(r.get("llm_judge_hallucination_rate", float("nan"))
           == r.get("llm_judge_hallucination_rate") for r in table):
        bars("llm_judge_hallucination_rate", "Hallucination rate (LLM-judge proxy)",
             "llm_judge_hallucination_rate.png")

    # precision/recall-style trade-off across strictness modes. Precision is 1 - the
    # (human, else LLM-judge) hallucination rate when labels exist, else validator pass rate.
    def precision(r: dict) -> tuple[float, str]:
        for key, label in (("hallucination_rate", "1 − human hallucination rate"),
                           ("llm_judge_hallucination_rate", "1 − LLM-judge hallucination rate")):
            v = r.get(key, float("nan"))
            if v == v:
                return 1 - v, label
        return r.get("validator_pass_rate", float("nan")), "validator pass rate"

    curve = [r for r in table if r["config"] == "B" or r["config"].startswith("C-")]
    ylabel = precision(curve[0])[1] if curve else ""
    pts: dict[tuple[float, float], list[str]] = {}
    for r in curve:
        pts.setdefault((round(r["retention"], 4), round(precision(r)[0], 4)), []).append(
            r["config"])
    ordered = sorted(pts.items())
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot([p[0] for p, _ in ordered], [p[1] for p, _ in ordered], "o-", color="#2f6f4f")
    for i, ((x, y), names) in enumerate(ordered):  # alternate above/below to avoid overlap
        ax.annotate(" = ".join(names), (x, y), textcoords="offset points",
                    xytext=(0, 9 if i % 2 == 0 else -16), ha="center", fontsize=9)
    ax.set_xlabel("retention: share of B's artifacts kept (recall proxy)")
    ax.set_ylabel(f"precision: {ylabel}")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strictness_tradeoff.png", dpi=160)
    plt.close(fig)

    strict = reasons.get("STRICT", {})
    if strict:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        keys = sorted(strict, key=strict.get, reverse=True)
        ax.barh(keys, [strict[k] for k in keys], color="#b5543c")
        ax.invert_yaxis()
        ax.set_title("Why extracted artifacts fail grounding (STRICT)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "drop_reasons.png", dpi=160)
        plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", choices=["ami", "samples"], default="ami")
    p.add_argument("--meetings", type=int, default=None, help="limit number of meetings")
    p.add_argument("--transcript", choices=["reference", "asr"], default="reference",
                   help="reference words (oracle ASR) or our own Whisper transcription")
    p.add_argument("--replay", action="store_true",
                   help="samples only: use recorded LLM responses (offline smoke test)")
    p.add_argument("--refresh", action="store_true", help="ignore cached extractions")
    p.add_argument("--model", default=None, help="override LLM_MODEL for this run")
    p.add_argument("--chunk-tokens", type=int, default=6000,
                   help="chunk size for config B (larger = fewer LLM calls)")
    p.add_argument("--label-n", type=int, default=150, help="statements in the labelling sheet")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if args.replay and args.dataset != "samples":
        sys.exit("--replay only works with --dataset samples")
    result = run(args)
    print("\nconfig        retention  pass_rate  unsupported_ent  halluc  action_f1  cost$")
    for r in result["table"]:
        print(f"{r['config']:<13} {fmt(r.get('retention')):>9} "
              f"{fmt(r.get('validator_pass_rate')):>10} "
              f"{fmt(r.get('unsupported_entity_rate')):>16} "
              f"{fmt(r.get('hallucination_rate')):>7} {fmt(r.get('action_f1', float('nan'))):>10} "
              f"{fmt(r.get('cost_usd', 0.0)):>6}")
    print(f"\n→ {RESULTS_DIR / 'ablation.csv'}")
    return result


if __name__ == "__main__":
    main()
