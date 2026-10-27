"""Load the three sample meetings (a 3-meeting series) so a fresh clone has something to show.

    cd backend && uv run python ../scripts/seed.py            # replayed LLM responses, offline
    cd backend && uv run python ../scripts/seed.py --live     # real extraction with LLM_MODEL

Runs each meeting through the real processing path (speaker naming, extraction,
grounding filter, storage, search indexing, accountability graph). If
scripts/make_sample_audio.py has been run, the audio is uploaded too and the
transcripts are aligned to it.
"""

from __future__ import annotations

import argparse
import json

from minuteai.api.processing import Services, process_meeting
from minuteai.config import get_settings
from minuteai.extract.extractor import Extractor
from minuteai.extract.llm_client import LLMClient
from minuteai.graph.service import GraphService
from minuteai.models.meeting import Meeting
from minuteai.samples import (
    SAMPLE_MEETINGS,
    SAMPLE_PARTICIPANTS,
    SAMPLE_SERIES_ID,
    ReplayProvider,
    sample_occurred_at,
    sample_transcript,
)
from minuteai.search.service import SearchService
from minuteai.storage.repository import MeetingRepository
from minuteai.storage.s3_store import audio_key
from minuteai.storage.schema import create_table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--live", action="store_true", help="call the configured LLM instead of replay")
    p.add_argument("--series", default=SAMPLE_SERIES_ID)
    args = p.parse_args()

    create_table()
    repo = MeetingRepository()
    repo.s3.ensure_bucket()
    samples_dir = get_settings().data_dir / "samples"
    search = SearchService(repo)
    graph = GraphService(repo)

    for mid, spec in SAMPLE_MEETINGS.items():
        timing_file = samples_dir / f"{mid}.timing.json"
        wav = samples_dir / f"{mid}.wav"
        timings = [tuple(t) for t in json.loads(timing_file.read_text())] \
            if timing_file.exists() else None
        transcript = sample_transcript(mid, timings)

        key = None
        if wav.exists():
            key = audio_key(mid, wav.name)
            repo.s3.put_audio(key, wav)
        repo.put_meeting(Meeting(
            meeting_id=mid, title=spec["title"], series_id=args.series,
            occurred_at=sample_occurred_at(mid), audio_s3_key=key,
            participants=SAMPLE_PARTICIPANTS, duration_sec=transcript.duration_sec,
            status="uploaded",
        ))
        repo.s3.put_transcript(transcript)  # cached transcript → processing skips ASR

        def extractor(mid=mid) -> Extractor:
            return Extractor() if args.live else Extractor(llm=LLMClient(ReplayProvider(mid)))

        services = Services(repo=repo, extractor=extractor)
        summary = process_meeting(services, mid)
        meeting = repo.require_meeting(mid)
        if summary is None:
            raise SystemExit(f"{mid}: processing failed: {meeting.error}")
        dropped = repo.get_dropped(mid)
        print(f"{mid}: {meeting.status} · {len(summary.abstract)} sentences, "
              f"{len(summary.decisions)} decisions, {len(summary.action_items)} action items, "
              f"{len(dropped)} filtered · audio={'yes' if key else 'no'}")

    n = search.rebuild()
    g = graph.build(args.series)
    print(f"search index: {n} entries · series {args.series!r}: {len(g.commitments)} commitments, "
          f"{len(g.reversals)} reversals")


if __name__ == "__main__":
    main()
