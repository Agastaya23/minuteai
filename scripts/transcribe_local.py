"""Transcribe a local audio file and print the first N utterances for inspection.

    cd backend && uv run python ../scripts/transcribe_local.py meeting.wav [--save out.json]

Checks to eyeball: utterance boundaries land at sensible places, speaker labels
switch at the right moments, word timestamps are present and increasing.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from minuteai.transcribe.audio import to_wav16k
from minuteai.transcribe.base import get_backend


def fmt(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("audio", type=Path)
    p.add_argument("--meeting-id", default="local")
    p.add_argument("--backend", default=None, help="whisper | aws_transcribe")
    p.add_argument("-n", type=int, default=20, help="utterances to print")
    p.add_argument("--save", type=Path, help="write the full Transcript JSON here")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        wav = to_wav16k(args.audio, Path(tmp) / "audio16k.wav")
        transcript = get_backend(args.backend).transcribe(wav, args.meeting_id)

    print(f"{transcript.model_version} · lang={transcript.language} · "
          f"{transcript.duration_sec:.1f}s · {len(transcript.utterances)} utterances · "
          f"speakers={transcript.speakers()}\n")
    monotonic = True
    last = -1.0
    for u in transcript.utterances:
        for w in u.words:
            monotonic &= w.start >= last
            last = w.start
    for u in transcript.utterances[: args.n]:
        print(f"[{u.utt_id:>3}] {u.speaker_label} ({fmt(u.start)}–{fmt(u.end)}): {u.text}")
    print(f"\nword timestamps monotonic: {monotonic}")
    if args.save:
        args.save.write_text(transcript.model_dump_json(indent=2))
        print(f"saved → {args.save}")


if __name__ == "__main__":
    main()
