"""ffmpeg helpers: normalize any input to 16 kHz mono PCM WAV."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from minuteai.transcribe.base import TranscriptionError

TARGET_RATE = 16_000


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise TranscriptionError("ffmpeg not found on PATH (brew install ffmpeg)")


def probe(path: Path) -> dict:
    _require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def is_16k_mono_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        info = probe(path)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    return bool(audio) and int(audio[0].get("sample_rate", 0)) == TARGET_RATE and int(
        audio[0].get("channels", 0)
    ) == 1 and audio[0].get("codec_name") == "pcm_s16le"


def to_wav16k(src: Path, dest: Path) -> Path:
    """Convert `src` to 16 kHz mono WAV at `dest` unless it already is one."""
    if is_16k_mono_wav(src):
        return src
    _require_ffmpeg()
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1",
         "-ar", str(TARGET_RATE), "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise TranscriptionError(f"ffmpeg failed: {proc.stderr.strip()[:500]}")
    return dest


def duration_sec(path: Path) -> float:
    return float(probe(path)["format"].get("duration", 0.0))
