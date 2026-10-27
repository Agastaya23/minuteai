# Evaluation harness

Everything runs from `backend/` with the backend environment (`uv sync --extra eval`).
Transcripts and LLM extractions are cached in `eval/cache/`, so re-running a runner
never re-transcribes or re-extracts.

## 1. Data

```bash
# AMI scenario meetings: annotations (~22 MB) + optional Mix-Headset audio
uv run python ../eval/datasets/ami.py --download                      # annotations only
uv run python ../eval/datasets/ami.py --download --audio --meetings ES2002a ES2002b ES2002c ES2002d
# QMSum (retrieval eval)
git clone https://github.com/Yale-LILY/QMSum ../eval/data/qmsum
```

`eval/data/` is git-ignored. AMI's ES20xx[a–d] are one team over four sessions, which is
also the input for the accountability-graph evaluation.

## 2. The ablation (headline experiment)

```bash
uv run python ../eval/runners/run_ablation.py --meetings 5                       # oracle transcript
uv run python ../eval/runners/run_ablation.py --meetings 5 --transcript asr      # our Whisper ASR
uv run python ../eval/runners/run_ablation.py --dataset samples --replay         # offline smoke test
```

| Config | What it is |
|---|---|
| A | whole transcript in one prompt, no chunking, no evidence requested |
| B | chunked, structured extraction with evidence, grounding filter **off** |
| C-STRICT / C-BALANCED / C-PERMISSIVE | B's output through the grounding filter |

Because C reuses B's extraction, **B → C isolates the filter**, and the three modes give
a retention vs. pass-rate trade-off curve (`figures/strictness_tradeoff.png`).

## 3. Human labelling → hallucination rate

The first run writes `eval/results/labelling_sheet.csv`: a stratified, shuffled sample of
statements from A, B and C-BALANCED with the cited utterances next to each one.

1. Fill `rater_1` with `supported`, `unsupported` or `contradicted` (check the full
   transcript for config A, which cites nothing).
2. Have a second person fill `rater_2` on an overlapping subset (≥ 50 items).
3. Re-run. The report then shows the hallucination rate per config and Cohen's κ.

The number to defend is **hallucination rate B vs. C**. If C isn't clearly lower,
tighten the entity-containment check first.

Without labels, `unsupported_entity_rate` (names/numbers/dates that occur nowhere in the
transcript) is an automatic lower bound on hallucination that also works for config A.

## 4. Everything else

```bash
uv run python ../eval/runners/run_all.py --meetings 5      # writes eval/results/report.md
```

| Metric module | Measures |
|---|---|
| `metrics/summarization.py` | ROUGE-1/2/L (+ BERTScore with `uv sync --extra bertscore`) |
| `metrics/action_items.py` | P/R/F1 by semantic matching; owner accuracy, abstention and wrong-assignment rates (separately) |
| `metrics/grounding.py` | labelling sheet, hallucination rate, retention, drop-reason taxonomy |
| `metrics/agreement.py` | Cohen's κ |
| `metrics/commitments.py` | lifecycle accuracy + confusion matrix vs `eval/data/commitment_labels.csv` |
| `metrics/search.py` | hit@k and MRR on QMSum's relevant spans |
| `metrics/cost.py` | tokens, $ and latency per meeting-hour from the LLM call logs |

Owner labels for AMI actions go in `eval/data/ami/action_owners.csv`
(`meeting_id,action_index,owner`; empty owner = nobody assigned).
