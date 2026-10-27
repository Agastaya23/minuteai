"""AMI Meeting Corpus loader.

Expected layout (see eval/README.md, or run `python eval/datasets/ami.py --download`):

    eval/data/ami/
      amicorpus/<ID>/audio/<ID>.Mix-Headset.wav
      annotations/                               (ami_public_manual_1.6.2.zip, unzipped)
        abstractive/<ID>.abssumm.xml             gold abstract / actions / decisions
        words/<ID>.<A-D>.words.xml               per-speaker words with timings
        segments/<ID>.<A-D>.segments.xml         gold speaker segmentation
    eval/data/ami/action_owners.csv              optional manual owner labels:
                                                  meeting_id,action_index,owner

Everything is normalized into our own Transcript model so the rest of the
harness is dataset-agnostic. The reference transcript built from the words
layer is an "oracle ASR" condition that isolates extraction from ASR errors.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable as a script

from eval.common import DATA_DIR, EvalMeeting  # noqa: E402
from minuteai.models.transcript import Transcript, Word  # noqa: E402
from minuteai.transcribe.diarization import SpeakerTurn, group_utterances  # noqa: E402

AMI_DIR = DATA_DIR / "ami"
ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AUDIO_URL = ("https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{id}/audio/"
             "{id}.Mix-Headset.wav")
# Scenario meetings: each ES20xx series is one team over four sessions (a–d).
ES_SERIES = [f"ES20{n:02d}" for n in range(2, 17)]
DEFAULT_MEETINGS = [f"{s}{x}" for s in ES_SERIES for x in "abcd"]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _sentences(root: ET.Element, section: str) -> list[str]:
    for el in root.iter():
        if _local(el.tag) == section:
            return [" ".join((s.text or "").split()) for s in el if _local(s.tag) == "sentence"
                    and (s.text or "").strip()]
    return []


def load_abstractive(meeting_id: str, root: Path = AMI_DIR) -> dict[str, list[str]]:
    path = root / "annotations" / "abstractive" / f"{meeting_id}.abssumm.xml"
    if not path.exists():
        return {}
    tree = ET.parse(path).getroot()
    return {k: _sentences(tree, k) for k in ("abstract", "actions", "decisions", "problems")}


def load_words(meeting_id: str, root: Path = AMI_DIR) -> list[tuple[Word, str]]:
    """All speakers' words, time-sorted, labelled SPEAKER_<letter index>."""
    words_dir = root / "annotations" / "words"
    out: list[tuple[Word, str]] = []
    for path in sorted(words_dir.glob(f"{meeting_id}.*.words.xml")):
        letter = path.name.split(".")[1]
        label = f"SPEAKER_{ord(letter) - ord('A'):02d}"
        for el in ET.parse(path).getroot():
            if _local(el.tag) != "w" or el.get("starttime") is None:
                continue
            text = (el.text or "").strip()
            if not text:
                continue
            start, end = float(el.get("starttime")), float(el.get("endtime") or el.get("starttime"))
            if el.get("punc") == "true" and out and out[-1][1] == label:
                prev = out[-1][0]
                out[-1] = (prev.model_copy(update={"text": prev.text + text}), label)
                continue
            out.append((Word(text=text, start=start, end=max(end, start + 0.01)), label))
    out.sort(key=lambda wl: wl[0].start)
    return out


def load_segments(meeting_id: str, root: Path = AMI_DIR) -> list[SpeakerTurn]:
    """Gold speaker segmentation (for diarization error analysis)."""
    turns = []
    for path in sorted((root / "annotations" / "segments").glob(f"{meeting_id}.*.segments.xml")):
        letter = path.name.split(".")[1]
        for el in ET.parse(path).getroot():
            if _local(el.tag) == "segment" and el.get("transcriber_start"):
                turns.append(SpeakerTurn(float(el.get("transcriber_start")),
                                         float(el.get("transcriber_end")),
                                         f"SPEAKER_{ord(letter) - ord('A'):02d}"))
    return sorted(turns, key=lambda t: t.start)


def reference_transcript(meeting_id: str, root: Path = AMI_DIR) -> Transcript | None:
    labeled = load_words(meeting_id, root)
    if not labeled:
        return None
    utts = group_utterances(labeled)
    return Transcript(meeting_id=meeting_id, utterances=utts, language="en",
                      duration_sec=utts[-1].end, backend="synthetic",
                      model_version="ami-reference-words")


def load_owner_labels(root: Path = AMI_DIR) -> dict[str, dict[int, str | None]]:
    path = root / "action_owners.csv"
    labels: dict[str, dict[int, str | None]] = {}
    if path.exists():
        with path.open() as f:
            for row in csv.DictReader(f):
                owner = (row.get("owner") or "").strip() or None
                labels.setdefault(row["meeting_id"], {})[int(row["action_index"])] = owner
    return labels


def load_meeting(meeting_id: str, root: Path = AMI_DIR) -> EvalMeeting:
    abs_ = load_abstractive(meeting_id, root)
    audio = root / "amicorpus" / meeting_id / "audio" / f"{meeting_id}.Mix-Headset.wav"
    return EvalMeeting(
        meeting_id=meeting_id,
        transcript=reference_transcript(meeting_id, root),
        audio_path=audio if audio.exists() else None,
        gold_abstract=abs_.get("abstract", []),
        gold_actions=abs_.get("actions", []),
        gold_decisions=abs_.get("decisions", []),
        gold_action_owners=load_owner_labels(root).get(meeting_id, {}),
        series_id=re.sub(r"[a-d]$", "", meeting_id),
        source="ami",
    )


def available_meetings(root: Path = AMI_DIR) -> list[str]:
    abs_dir = root / "annotations" / "abstractive"
    if not abs_dir.exists():
        return []
    ids = sorted(p.name.split(".")[0] for p in abs_dir.glob("ES20*.abssumm.xml"))
    return [i for i in ids if (root / "annotations" / "words" / f"{i}.A.words.xml").exists()]


def load(n: int | None = None, meeting_ids: list[str] | None = None,
         root: Path = AMI_DIR) -> list[EvalMeeting]:
    ids = meeting_ids or available_meetings(root)
    return [load_meeting(i, root) for i in (ids[:n] if n else ids)]


def download(meeting_ids: list[str], with_audio: bool, root: Path = AMI_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ann = root / "annotations"
    if not (ann / "abstractive").exists():
        zip_path = root / "ami_public_manual_1.6.2.zip"
        if not zip_path.exists():
            print(f"downloading annotations → {zip_path}")
            urllib.request.urlretrieve(ANNOTATIONS_URL, zip_path)  # noqa: S310
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ann)
    if with_audio:
        for mid in meeting_ids:
            dest = root / "amicorpus" / mid / "audio" / f"{mid}.Mix-Headset.wav"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"downloading {mid} audio")
                urllib.request.urlretrieve(AUDIO_URL.format(id=mid), dest)  # noqa: S310


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download / inspect AMI")
    p.add_argument("--download", action="store_true")
    p.add_argument("--audio", action="store_true", help="also download Mix-Headset audio")
    p.add_argument("--meetings", nargs="*", default=["ES2002a", "ES2002b", "ES2002c", "ES2002d"])
    args = p.parse_args()
    if args.download:
        download(args.meetings, args.audio)
    for m in load(meeting_ids=args.meetings):
        t = m.transcript
        print(f"{m.meeting_id}: {len(t.utterances) if t else 0} utterances, "
              f"{len(m.gold_abstract)} abstract, {len(m.gold_actions)} actions, "
              f"{len(m.gold_decisions)} decisions, audio={'yes' if m.audio_path else 'no'}")
