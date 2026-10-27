"""Run extraction (and optionally the grounding filter) over a cached transcript.

    cd backend && uv run python ../scripts/extract_local.py <meeting_id> [--filter]
    cd backend && uv run python ../scripts/extract_local.py --file transcript.json
    cd backend && uv run python ../scripts/extract_local.py --sample demo-sprint-planning --replay

Check: are the cited utt_ids real, and do those utterances say what is claimed?
"""

from __future__ import annotations

import argparse
from pathlib import Path

from minuteai.extract.extractor import Extractor
from minuteai.extract.llm_client import LLMClient
from minuteai.models.artifacts import MeetingSummary
from minuteai.models.transcript import Transcript


def load_transcript(args) -> Transcript:
    if args.file:
        return Transcript.model_validate_json(Path(args.file).read_text())
    if args.sample:
        from minuteai.samples import sample_transcript

        return sample_transcript(args.sample)
    from minuteai.storage import S3Store

    t = S3Store().get_transcript(args.meeting_id)
    if t is None:
        raise SystemExit(f"no cached transcript for {args.meeting_id!r}")
    return t


def show(summary: MeetingSummary, transcript: Transcript) -> None:
    def ev(evidence) -> str:
        return ", ".join(f"[{e.utt_id}] “{e.quote}”" for e in evidence) or "(none)"

    print(f"\n== Abstract ({len(summary.abstract)}) ==")
    for s in summary.abstract:
        print(f"• {s.text}\n    ↳ {ev(s.evidence)}{'  ⚑ ' + ','.join(s.flags) if s.flags else ''}")
    print(f"\n== Topics ({len(summary.topics)}) ==")
    for t in summary.topics:
        print(f"• {t.title}  [utt {t.start_utt}–{t.end_utt}]")
    print(f"\n== Decisions ({len(summary.decisions)}) ==")
    for d in summary.decisions:
        flags = f"  ⚑ {','.join(d.flags)}" if d.flags else ""
        print(f"• [{d.status}] {d.statement} (conf {d.confidence:.2f}, {d.participants})\n"
              f"    ↳ {ev(d.evidence)}{flags}")
    print(f"\n== Action items ({len(summary.action_items)}) ==")
    for a in summary.action_items:
        owner = f"{a.owner} ({a.owner_resolution_method}, {a.owner_confidence:.2f})" \
            if a.owner else "UNASSIGNED"
        due = f" due {a.due_date} ({a.due_raw})" if a.due_raw else ""
        flags = f"  ⚑ {','.join(a.flags)}" if a.flags else ""
        print(f"• {a.description} — {owner}{due}\n    ↳ {ev(a.evidence)}{flags}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("meeting_id", nargs="?")
    p.add_argument("--file")
    p.add_argument("--sample")
    p.add_argument("--replay", action="store_true", help="use recorded sample LLM responses")
    p.add_argument("--filter", action="store_true", help="apply the grounding filter")
    p.add_argument("--mode", default=None, help="STRICT | BALANCED | PERMISSIVE")
    args = p.parse_args()
    if not (args.meeting_id or args.file or args.sample):
        p.error("give a meeting_id, --file or --sample")

    transcript = load_transcript(args)
    llm = None
    if args.replay:
        from minuteai.samples import ReplayProvider

        llm = LLMClient(ReplayProvider(args.sample or transcript.meeting_id))
    summary = Extractor(llm=llm).extract_meeting(transcript)
    if args.filter:
        from minuteai.ground.filter import GroundingFilter

        summary, dropped = GroundingFilter(transcript, mode=args.mode).apply(summary)
        show(summary, transcript)
        print(f"\n== Dropped by grounding filter ({len(dropped)}) ==")
        for d in dropped:
            print(f"✗ [{d.artifact_type}] {d.text}\n    ↳ "
                  + "; ".join(f"{f.reason}: {f.detail}" for f in d.failures))
    else:
        show(summary, transcript)


if __name__ == "__main__":
    main()
