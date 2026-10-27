"""Generate spoken audio for the sample meetings (macOS `say` + ffmpeg).

    cd backend && uv run python ../scripts/make_sample_audio.py

Writes data/samples/<meeting_id>.wav (16 kHz mono) and <meeting_id>.timing.json
with each utterance's measured start/end, so seeded transcripts line up with the
audio and evidence clicks seek to the right moment. Each speaker gets a voice.
The WAVs are also realistic multi-speaker input for scripts/transcribe_local.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from minuteai.config import get_settings
from minuteai.samples import SAMPLE_MEETINGS

VOICES = {"SPEAKER_00": "Tara", "SPEAKER_01": "Rishi", "SPEAKER_02": "Samantha"}
FALLBACK_VOICE = "Samantha"
GAP_SEC = 0.6
RATE = 16_000


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def render(meeting_id: str, out_dir: Path, voices: dict[str, str]) -> Path:
    lines = SAMPLE_MEETINGS[meeting_id]["lines"]
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        silence = tmpdir / "gap.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"anullsrc=r={RATE}:cl=mono", "-t", str(GAP_SEC), "-c:a", "pcm_s16le",
                        str(silence)], check=True)
        parts, timings, t = [], [], 0.0
        for i, (speaker, text) in enumerate(lines):
            aiff, wav = tmpdir / f"{i}.aiff", tmpdir / f"{i}.wav"
            subprocess.run(["say", "-v", voices.get(speaker, FALLBACK_VOICE), "-o", str(aiff),
                            text], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ac", "1",
                            "-ar", str(RATE), "-c:a", "pcm_s16le", str(wav)], check=True)
            d = duration(wav)
            timings.append((round(t, 3), round(t + d, 3)))
            parts += [wav, silence]
            t += d + GAP_SEC
        listing = tmpdir / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        out = out_dir / f"{meeting_id}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-c:a", "pcm_s16le", str(out)], check=True)
    (out_dir / f"{meeting_id}.timing.json").write_text(json.dumps(timings))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        sys.exit("needs macOS `say` and ffmpeg")
    out_dir = args.out or get_settings().data_path("samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    for mid in SAMPLE_MEETINGS:
        path = render(mid, out_dir, VOICES)
        print(f"{mid}: {path} ({duration(path):.1f}s)")


if __name__ == "__main__":
    main()
