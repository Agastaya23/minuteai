import csv
import math

import pytest

from eval.metrics import (
    action_items,
    agreement,
    commitments,
    cost,
    grounding,
    search,
    summarization,
)
from minuteai.models.artifacts import ActionItem, GroundedSentence, MeetingSummary
from minuteai.samples import sample_transcript
from minuteai.search.embedder import HashingEmbedder

EMB = HashingEmbedder()


def test_rouge_identical_and_disjoint():
    assert summarization.rouge("the cat sat", "the cat sat")["rouge1"] == pytest.approx(1.0)
    assert summarization.rouge("alpha beta", "gamma delta")["rouge1"] == 0.0
    agg = summarization.evaluate([("the cat sat", "the cat sat"), ("a", "b")])
    assert agg["n"] == 2 and 0 < agg["rouge1"] < 1


def _item(desc, owner=None):
    return ActionItem(item_id=desc[:4], description=desc, owner=owner,
                      owner_resolution_method="self_commitment" if owner else "unresolved")


def test_action_items_prf_and_owner_rates():
    pred = [_item("Write the migration script", "Rahul"),
            _item("Set up the staging database", None),
            _item("Update the on-call rotation", "Rahul"),
            _item("Order pizza for the team", "Priya")]
    gold = ["Write the migration script by Friday", "Set up staging database on port 5432",
            "Update the on-call rotation"]
    owners = {0: "Rahul", 1: "Ananya", 2: "Priya"}
    r = action_items.evaluate(pred, gold, owners, EMB)
    assert r["tp"] == 3
    assert r["precision"] == pytest.approx(0.75) and r["recall"] == pytest.approx(1.0)
    assert r["owner_accuracy"] == pytest.approx(1 / 3)
    assert r["abstention_rate"] == pytest.approx(1 / 3)
    assert r["wrong_assignment_rate"] == pytest.approx(1 / 3)


def test_action_items_empty():
    assert action_items.evaluate([], ["x"])["recall"] == 0.0


def test_kappa():
    assert agreement.cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    k = agreement.cohens_kappa(["s", "s", "u", "u"], ["s", "u", "s", "u"])
    assert k == pytest.approx(0.0)
    assert agreement.interpret(0.7) == "substantial"
    with pytest.raises(ValueError):
        agreement.cohens_kappa(["a"], [])


def test_commitment_confusion():
    r = commitments.evaluate({"c1": "FULFILLED", "c2": "SLIPPED", "c3": "COMMITTED"},
                             {"c1": "FULFILLED", "c2": "FULFILLED"})
    assert r["n"] == 2 and r["accuracy"] == 0.5
    assert r["confusion"]["FULFILLED"]["SLIPPED"] == 1
    assert r["unlabelled_predictions"] == 1


def test_search_metrics():
    r = search.evaluate([([(0, 4), (5, 9)], [(6, 6)]), ([(10, 12)], [(0, 1)])], ks=(1, 2))
    assert r["hit@1"] == 0.0 and r["hit@2"] == 0.5 and r["mrr"] == pytest.approx(0.25)


def test_cost_summary():
    calls = [{"model": "claude-haiku-5-5", "meeting_id": "m", "input_tokens": 1_000_000,
              "output_tokens": 100_000, "latency_ms": 1500, "success": True, "retry_count": 1,
              "purpose": "summary"}]
    s = cost.summarize(calls, {"m": 0.5})
    assert s["totals"]["cost_usd"] == pytest.approx(0.10 + 0.05)
    assert s["per_meeting_hour"]["cost_usd"] == pytest.approx(0.30)
    assert s["retries"] == 1


def test_labelling_sheet_roundtrip(tmp_path):
    t = sample_transcript()
    from minuteai.models.evidence import EvidenceSpan

    u = t.utterances[6]
    ev = EvidenceSpan(utt_id=6, start_sec=u.start, end_sec=u.end, speaker_label=u.speaker_label,
                      quote="move the payments service")
    s = MeetingSummary(meeting_id=t.meeting_id,
                       abstract=[GroundedSentence(text=f"Sentence {i}", evidence=[ev])
                                 for i in range(6)],
                       action_items=[_item("Write the migration script", "Rahul")])
    items = grounding.statements("B", s) + grounding.statements("C-BALANCED", s)
    sample = grounding.stratified_sample(items, 6)
    assert {x.config for x in sample} == {"B", "C-BALANCED"}
    assert any(x.artifact_type == "action_item" for x in sample)
    path = grounding.write_labelling_sheet(sample, {t.meeting_id: t}, tmp_path / "sheet.csv")
    rows = list(csv.DictReader(path.open()))
    assert all("Priya:" in r["cited_evidence"] or "(no evidence" in r["cited_evidence"]
               for r in rows)
    for r, label in zip(rows, ["supported", "unsupported", "contradicted", "supported",
                               "supported", "supported"], strict=True):
        r["rater_1"] = label
    rates = grounding.hallucination_rates(rows)
    total = sum(v["n"] for v in rates.values())
    bad = sum(v["unsupported"] + v["contradicted"] for v in rates.values())
    assert total == 6 and bad == 2


def test_entity_support_proxy():
    t = sample_transcript()
    good = MeetingSummary(meeting_id=t.meeting_id,
                          abstract=[GroundedSentence(text="Staging uses port 5432.")])
    bad = MeetingSummary(meeting_id=t.meeting_id,
                         abstract=[GroundedSentence(text="Staging uses port 9999 on Kubernetes.")])
    assert grounding.transcript_entity_support(good, t)["entity_supported_rate"] == 1.0
    assert grounding.transcript_entity_support(bad, t)["entity_supported_rate"] == 0.0
    assert math.isnan(grounding.transcript_entity_support(
        MeetingSummary(meeting_id="x"), t)["entity_supported_rate"])


def test_ami_parsers(tmp_path):
    from eval.datasets import ami

    ann = tmp_path / "annotations"
    (ann / "abstractive").mkdir(parents=True)
    (ann / "words").mkdir()
    ns = 'xmlns:nite="http://nite.sourceforge.net/"'
    (ann / "abstractive" / "ES2002a.abssumm.xml").write_text(
        f'<nite:root {ns}><abstract><sentence nite:id="1">The team met.</sentence></abstract>'
        '<actions><sentence nite:id="2">Design the remote.</sentence></actions>'
        '<decisions><sentence nite:id="3">Use a  battery.</sentence></decisions></nite:root>')
    for letter, words in (("A", [("Hello", 0.0, 0.4), (".", 0.4, 0.4)]),
                          ("B", [("Hi", 1.0, 1.2), ("there", 1.25, 1.5)])):
        punc = ' punc="true"'
        body = "".join(
            f'<w nite:id="{letter}{i}" starttime="{s}" endtime="{e}"'
            f'{punc if w == "." else ""}>{w}</w>'
            for i, (w, s, e) in enumerate(words)
        )
        (ann / "words" / f"ES2002a.{letter}.words.xml").write_text(
            f"<nite:root {ns}>{body}</nite:root>")
    m = ami.load_meeting("ES2002a", tmp_path)
    assert m.gold_abstract == ["The team met."]
    assert m.gold_actions == ["Design the remote."] and m.gold_decisions == ["Use a battery."]
    assert [(u.speaker_label, u.text) for u in m.transcript.utterances] == [
        ("SPEAKER_00", "Hello."), ("SPEAKER_01", "Hi there")]
    assert ami.available_meetings(tmp_path) == ["ES2002a"]
    assert m.series_id == "ES2002"


def test_qmsum_loader(tmp_path):
    import json

    from eval.datasets import qmsum

    folder = tmp_path / "data" / "ALL" / "test"
    folder.mkdir(parents=True)
    (folder / "m1.json").write_text(json.dumps({
        "meeting_transcripts": [{"speaker": "A", "content": "hello there"},
                                {"speaker": "B", "content": "budget is tight"}],
        "specific_query_list": [{"query": "budget?", "answer": "tight",
                                 "relevant_text_span": [["1", "1"]]}],
        "general_query_list": [{"query": "summarize", "answer": "x"}],
    }))
    [m] = qmsum.load(root=tmp_path)
    assert m.transcript.utterances[1].speaker_name == "B"
    assert m.queries[0].relevant_spans == [(1, 1)]


def test_derived_filter_rates():
    rows = [{"config": "B", "meeting_id": "m", "text": t, "j": lab}
            for t, lab in (("a", "supported"), ("b", "unsupported"), ("c", "contradicted"),
                           ("d", "supported"))]
    rows.append({"config": "A", "meeting_id": "m", "text": "a", "j": "unsupported"})
    out = grounding.derived_filter_rates(rows, "j", {"C-STRICT": {("m", "a"), ("m", "d")},
                                                     "C-PERMISSIVE": {("m", t) for t in "abcd"}})
    assert out["C-STRICT"]["hallucination_rate"] == 0.0 and out["C-STRICT"]["n"] == 2
    assert out["C-PERMISSIVE"]["hallucination_rate"] == 0.5
