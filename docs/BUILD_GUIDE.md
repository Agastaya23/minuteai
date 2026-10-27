# MinuteAI — Claude Code Build Guide

A sequenced, copy-paste implementation plan. Each phase has prerequisites, the exact prompt to give Claude Code, and a verification step. Do not skip the verification steps — they are what stop you discovering in week 8 that phase 3 was wrong.

---

## Part 0 — Before you open Claude Code

### 0.1 Accounts and keys

| Thing | Why | Cost | Action |
|---|---|---|---|
| AWS account | S3, DynamoDB | Free tier covers this project easily | Create IAM user `minuteai-dev`, attach `AmazonS3FullAccess` + `AmazonDynamoDBFullAccess`, generate access keys |
| LLM API key | Extraction + narration | ₹300–800 for the whole project | Anthropic, OpenAI, or Google AI Studio (Gemini Flash is cheapest and fine here) |
| HuggingFace token | `pyannote.audio` speaker diarization | Free | Create token, then accept the user agreement on `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` model pages — **it will fail silently-ish without this** |
| AMI Meeting Corpus | Evaluation data (audio + gold minutes) | Free | `groups.inf.ed.ac.uk/ami/download/` — download the ES2002–ES2016 scenario series |
| QMSum | Evaluation data (query-based) | Free | `github.com/Yale-LILY/QMSum` |

### 0.2 Local tooling

```
Python 3.11+        (3.12 fine; avoid 3.13 for pyannote compatibility)
Node 20+
ffmpeg              (Whisper requires it — brew install ffmpeg / apt install ffmpeg)
Docker              (DynamoDB Local)
git
AWS CLI v2          (aws configure with your keys)
```

Verify before starting:
```bash
python --version && node --version && ffmpeg -version | head -1 && docker --version
aws sts get-caller-identity
```

### 0.3 Decide these now (Claude Code will ask otherwise)

| Decision | Recommendation | Why |
|---|---|---|
| Transcription | **faster-whisper locally**, AWS Transcribe as a pluggable second adapter | Free, offline, gives word-level timestamps which the grounding feature *requires*. Transcribe costs per minute and you'll run hundreds of test transcriptions. |
| Diarization | `pyannote.audio` 3.1 | Needed for speaker attribution and owner assignment |
| Backend | **FastAPI (Python)** | Whisper and pyannote are Python; a Node backend would force an awkward split |
| Frontend | Next.js 15, App Router, TypeScript, Tailwind, shadcn/ui | As specified in your abstract |
| Primary DB | DynamoDB (single-table design) | As specified |
| Vector search | FAISS index persisted to S3 | DynamoDB has no vector search. OpenSearch costs real money. FAISS is a file — simple and defensible. |
| LLM | Gemini 2.0 Flash or Claude Haiku | Cheap, both support structured JSON output |
| Local dev storage | DynamoDB Local in Docker + real S3 | DynamoDB Local saves you from scanning costs during iteration |

---

## Part 1 — Project scaffold and CLAUDE.md

### Create the repo yourself (don't make Claude Code do this)

```bash
mkdir minuteai && cd minuteai && git init
mkdir -p backend frontend eval docs scripts data
```

### Prompt 1 — Scaffold

> Create the project scaffold for MinuteAI, a meeting intelligence platform.
>
> Monorepo layout:
> - `backend/` — FastAPI app, Python 3.11, managed with uv (fall back to venv + pip if uv isn't installed)
> - `frontend/` — Next.js 15 App Router, TypeScript, Tailwind, shadcn/ui
> - `eval/` — evaluation harness, Python
> - `scripts/` — one-off setup and utility scripts
> - `docs/` — design docs
>
> In `backend/`, set up: FastAPI, pydantic v2, boto3, python-dotenv, pytest, ruff. Create packages `minuteai/{models,storage,transcribe,extract,ground,graph,search,api}` each with `__init__.py`.
>
> Create `.env.example` with placeholders for AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET, DDB_TABLE, DDB_ENDPOINT_URL, LLM_API_KEY, LLM_MODEL, HF_TOKEN.
>
> Create `docker-compose.yml` running DynamoDB Local on port 8000.
>
> Create a `Makefile` with targets: `dev-backend`, `dev-frontend`, `test`, `lint`, `ddb-up`, `ddb-down`.
>
> Add a `.gitignore` covering Python, Node, `.env`, `data/`, `*.db`, model caches.
>
> Don't implement any business logic yet. Scaffold only.

### Prompt 2 — CLAUDE.md (do this immediately after)

> Create a `CLAUDE.md` at the repo root with the following content, adapted to the actual scaffold you just built. This file is read at the start of every session, so keep it under 150 lines and make it operational, not descriptive.
>
> Include:
> 1. One-paragraph project description.
> 2. The architecture pipeline as a single diagram: audio → S3 → transcription (utterances with word-level timestamps) → LLM extraction → grounding validation → DynamoDB → search index → Next.js.
> 3. **Hard rules** section, stated as non-negotiable:
>    - Every extracted artifact (summary sentence, decision, action item) MUST carry `evidence: list[EvidenceSpan]` pointing to real utterance IDs and timestamps. No exceptions.
>    - The LLM never determines structure, only content. Structure comes from deterministic Python.
>    - All LLM calls go through `minuteai/extract/llm_client.py`. No direct SDK calls anywhere else.
>    - All LLM responses are parsed into pydantic models. Never consume raw text.
>    - All storage access goes through `minuteai/storage/`. No boto3 calls elsewhere.
>    - Transcription backends are adapters behind a common interface. No backend-specific types leak out.
> 4. Commands table: how to run backend, frontend, tests, lint, DynamoDB Local.
> 5. Code conventions: pydantic v2 models for all data crossing a module boundary; type hints everywhere; pytest with fixtures in `conftest.py`; no bare `except`.
> 6. A "where things live" table mapping each package to its responsibility.

**Verify:** `make dev-backend` serves FastAPI on :8080, `make dev-frontend` serves Next.js on :3000, `make ddb-up` starts DynamoDB Local, `make test` runs (zero tests is fine).

**Commit.** Commit after every phase from here on.

---

## Part 2 — Data models first

This phase is the highest-leverage one in the entire project. Get the models right and everything downstream is mechanical. Get them wrong and you will refactor for a week.

### Prompt 3 — Core models

> Implement the core data models in `backend/minuteai/models/`. Use pydantic v2. These are the contract for the entire system — be precise.
>
> `transcript.py`:
> - `Word`: text, start (float seconds), end, confidence
> - `Utterance`: utt_id (int, sequential from 0), speaker_label (e.g. "SPEAKER_01"), speaker_name (str | None — resolved later), text, start, end, words (list[Word])
> - `Transcript`: meeting_id, utterances (list[Utterance]), language, duration_sec, backend ("whisper" | "aws_transcribe"), model_version
> - Helper methods on Transcript: `window(start_utt, end_utt)` returns the utterance slice; `text_between(start_sec, end_sec)`; `total_words()`
>
> `evidence.py`:
> - `EvidenceSpan`: utt_id (int), start_sec, end_sec, speaker_label, quote (the exact substring from that utterance supporting the claim, max 200 chars)
> - This type is the backbone of the whole system. Every extracted artifact references it.
>
> `artifacts.py`:
> - `GroundedSentence`: text, evidence (list[EvidenceSpan]), validated (bool, default False)
> - `Decision`: decision_id, statement, rationale (str | None), participants (list[str]), evidence, confidence (float 0–1), status (Literal["proposed","agreed","deferred","reversed"])
> - `ActionItem`: item_id, description, owner (str | None), owner_confidence (float), owner_resolution_method (Literal["explicit_name","self_commitment","assigned_by_other","unresolved"]), due_date (date | None), due_raw (the original phrase e.g. "by end of next week"), evidence, confidence
> - `MeetingSummary`: meeting_id, abstract (list[GroundedSentence]), topics (list[Topic]), decisions, action_items, participants, generated_at, model_version
> - `Topic`: topic_id, title, start_utt, end_utt, summary (list[GroundedSentence])
>
> `meeting.py`:
> - `Meeting`: meeting_id, title, series_id (str | None — groups recurring meetings), occurred_at (datetime), duration_sec, audio_s3_key, participants, status (Literal["uploaded","transcribing","extracting","ready","failed"]), created_at, error (str | None)
>
> Critical: `owner` on ActionItem must be nullable and `owner_resolution_method` must include `"unresolved"`. The system must be able to say "nobody was clearly assigned" rather than guessing.
>
> Write pytest tests in `backend/tests/test_models.py` covering serialization round-trips and the Transcript helper methods. Include a fixture that builds a small synthetic 20-utterance transcript — other phases will reuse it.

**Verify:** `make test` passes. Read `evidence.py` and `artifacts.py` yourself. These two files define what your project is; you need to understand them cold for the viva.

---

## Part 3 — Storage layer

### Prompt 4 — DynamoDB single-table design + S3

> Implement the storage layer in `backend/minuteai/storage/`.
>
> **Single-table DynamoDB design.** Table `minuteai`, with:
> - PK/SK composite key
> - GSI1 (GSI1PK/GSI1SK) for series-based and status-based access
>
> Access patterns to support:
> 1. Get meeting by id → `PK=MEETING#<id>, SK=META`
> 2. Get transcript for meeting → `PK=MEETING#<id>, SK=TRANSCRIPT`
> 3. Get summary for meeting → `PK=MEETING#<id>, SK=SUMMARY`
> 4. List action items for meeting → `PK=MEETING#<id>, SK begins_with ACTION#`
> 5. List decisions for meeting → `PK=MEETING#<id>, SK begins_with DECISION#`
> 6. List all meetings in a series, newest first → `GSI1PK=SERIES#<id>, GSI1SK=<occurred_at ISO>`
> 7. List all action items owned by a person across meetings → `GSI1PK=OWNER#<normalized_name>, GSI1SK=<occurred_at>#<item_id>`
> 8. List meetings by status → `GSI1PK=STATUS#<status>, GSI1SK=<created_at>`
>
> Write `schema.py` documenting every item type's key structure as a table in a module docstring.
>
> `repository.py` — a `MeetingRepository` class with typed methods returning pydantic models, never raw dicts: `put_meeting`, `get_meeting`, `update_status`, `put_transcript`, `get_transcript`, `put_summary`, `get_summary`, `list_series`, `list_by_owner`, `list_by_status`. Large transcripts exceed the 400KB DynamoDB item limit — store the full transcript JSON in S3 at `transcripts/<meeting_id>.json` and keep only a pointer plus metadata in DynamoDB. Make this transparent to the caller.
>
> `s3_store.py` — put/get audio, put/get transcript JSON, generate presigned upload and download URLs.
>
> `scripts/create_table.py` — idempotent table + GSI creation, works against both DynamoDB Local and real AWS based on `DDB_ENDPOINT_URL`.
>
> Tests using `moto` to mock AWS. Cover every access pattern above.

**Verify:** `make ddb-up && python scripts/create_table.py && make test`. Access patterns 7 and 8 are what the cross-meeting features need later — confirm they actually work now.

---

## Part 4 — Transcription pipeline

### Prompt 5 — Whisper + diarization

> Implement transcription in `backend/minuteai/transcribe/`.
>
> `base.py` — abstract `TranscriptionBackend` with `transcribe(audio_path: Path, meeting_id: str) -> Transcript`.
>
> `whisper_backend.py` — uses `faster-whisper`. Must set `word_timestamps=True`; word-level timing is required downstream and the system is useless without it. Default model `small`, configurable via env. Handle long audio without loading everything into memory at once.
>
> `diarization.py` — uses `pyannote.audio` 3.1 to produce speaker turns, then merges with Whisper output by assigning each word to the speaker turn with maximum temporal overlap. Group consecutive same-speaker words into `Utterance` objects, splitting on pauses longer than 1.5 seconds or when the speaker changes. Assign sequential `utt_id` starting at 0.
>
> `aws_backend.py` — AWS Transcribe adapter with `ShowSpeakerLabels=True`, normalizing its output into the identical `Transcript` shape. Implement it, but do not make it the default.
>
> `pipeline.py` — `transcribe_meeting(audio_s3_key, meeting_id, backend) -> Transcript`: downloads from S3 to a temp file, converts to 16kHz mono WAV with ffmpeg if needed, runs the backend, persists via the storage layer, updates meeting status.
>
> Cache transcripts: if `transcripts/<meeting_id>.json` already exists in S3, return it unless `force=True`. You will re-run this pipeline dozens of times during development and re-transcribing each time wastes hours.
>
> Add `scripts/transcribe_local.py` — a CLI that transcribes a local audio file and prints the first 20 utterances, for manual inspection.

**Verify:** Take a 5-minute multi-speaker audio clip (an AMI meeting, or record yourself and a friend). Run `scripts/transcribe_local.py`. Check: do utterance boundaries land in sensible places? Do speaker labels switch at the right moments? Are word timestamps present and monotonically increasing?

Diarization is the flakiest part of this stack. If speaker assignment is visibly wrong, fix it now — every downstream feature inherits the error.

---

## Part 5 — LLM extraction

### Prompt 6 — LLM client

> Implement `backend/minuteai/extract/llm_client.py`.
>
> A single `LLMClient` class, the only place in the codebase that talks to an LLM provider:
> - `complete_structured(prompt: str, response_model: Type[BaseModel], max_retries: int = 2) -> BaseModel`
> - Uses the provider's JSON/structured output mode
> - On parse failure, retries once with the validation error appended to the prompt, then raises `LLMParseError`
> - Logs every call to `data/llm_calls.jsonl`: timestamp, prompt hash, model, input tokens, output tokens, latency, success/failure, retry count. This log is your cost and latency data for the evaluation chapter — do not skip it.
> - Supports Gemini and Anthropic behind one interface, selected by the `LLM_MODEL` env var
> - Exponential backoff on rate limits
>
> Tests mock the provider; assert retry behaviour and that the call log is written correctly.

### Prompt 7 — Chunking and extraction

> Implement extraction in `backend/minuteai/extract/`.
>
> `chunker.py` — `chunk_transcript(transcript, max_tokens=6000, overlap_utts=3) -> list[TranscriptChunk]`. Prefer splitting at topic boundaries over fixed sizes: detect boundaries using cosine distance between embeddings of sliding utterance windows, with a fallback to token-count splits. Every chunk must retain the real `utt_id` values — never renumber. `TranscriptChunk` carries chunk_id, start_utt, end_utt, utterances.
>
> `prompts.py` — prompt templates as module constants, each documented with what it returns.
>
> `extractor.py`:
> - `extract_topics(transcript) -> list[Topic]`
> - `extract_summary(chunk) -> list[GroundedSentence]`
> - `extract_decisions(chunk) -> list[Decision]`
> - `extract_action_items(chunk) -> list[ActionItem]`
> - `extract_meeting(transcript) -> MeetingSummary` — orchestrates the above and merges chunk-level results, deduplicating near-identical items across overlapping chunks
>
> **Prompt design rules, applied to every extraction prompt:**
> - Utterances are presented to the model as `[utt_id] SPEAKER_XX (mm:ss): text`
> - The prompt explicitly instructs: every returned object must include an `evidence` array listing the `utt_id` values that support it, and a short verbatim `quote` from each
> - The prompt instructs the model to return an empty list rather than inventing content when nothing qualifies
> - For action items, the prompt instructs: if no owner is clearly identifiable, set `owner` to null and `owner_resolution_method` to `"unresolved"` — do not guess
> - Response schema enforced through `complete_structured`
>
> Write `scripts/extract_local.py` — runs extraction over a cached transcript and pretty-prints the result.

**Verify:** Run extraction on your test meeting. Check specifically: are the `utt_id` values in the evidence arrays real and do they actually contain what's claimed? Some will be wrong. That's expected — Phase 6 is what catches them, and the fact that they're wrong is your motivation section.

---

## Part 6 — Grounding validator *(Novelty 1)*

### Prompt 8

> Implement `backend/minuteai/ground/validator.py`. This is the core novel contribution — implement it carefully and make every check independently testable.
>
> `GroundingValidator` takes a `Transcript` and validates any artifact carrying `evidence: list[EvidenceSpan]`. Checks, in order:
>
> 1. **Existence** — every referenced `utt_id` exists in the transcript. Fail reason: `UTT_NOT_FOUND`.
> 2. **Quote fidelity** — each `EvidenceSpan.quote` appears in the referenced utterance's text, compared after normalization (lowercase, collapse whitespace, strip punctuation) with fuzzy matching at a ratio ≥ 0.85 to tolerate minor LLM paraphrase. Fail: `QUOTE_NOT_FOUND`.
> 3. **Timestamp consistency** — the span's start/end are within the referenced utterance's bounds, with 0.5s tolerance. Fail: `TIMESTAMP_MISMATCH`.
> 4. **Speaker consistency** — if the artifact names a speaker (an action item owner, a decision participant), that speaker label must appear among the referenced utterances. Fail: `SPEAKER_MISMATCH`.
> 5. **Entity containment** — extract named entities, numbers, and dates from the artifact's own text via regex and spaCy; each must appear in the concatenated text of the referenced utterances (dates allowed to be resolved forms of relative phrases present there). Fail: `ENTITY_NOT_SUPPORTED`.
> 6. **Non-empty evidence** — zero evidence spans is an automatic fail. Fail: `NO_EVIDENCE`.
>
> Return a `ValidationResult`: passed (bool), failures (list of (check_name, reason, detail)), confidence (float, computed as the fraction of checks passed weighted by severity).
>
> `filter.py` — `GroundingFilter` applies the validator across a whole `MeetingSummary`, returning `(kept: MeetingSummary, dropped: list[DroppedArtifact])`. Every dropped artifact is logged to `data/dropped_artifacts.jsonl` with its full text, evidence, and failure reasons. **This log is a research artifact** — the taxonomy of why generation fails grounding is a finding for the report.
>
> Make strictness configurable: `STRICT` (any failure drops), `BALANCED` (drops on checks 1, 2, 6; flags the rest), `PERMISSIVE` (flags only, drops nothing). Default `BALANCED`. The three modes let you report a precision/recall tradeoff curve instead of a single point.
>
> Tests must cover each failure mode individually with a hand-built failing artifact, plus one artifact that passes everything.

**Verify:** Re-run extraction on your test meeting with the filter on. Read `dropped_artifacts.jsonl`. If nothing is dropped, your checks are too loose — hand-craft a hallucinated artifact (cite utterance 999, or claim a port number not in the transcript) and confirm it gets caught.

---

## Part 7 — API and frontend (ship the core here)

### Prompt 9 — FastAPI

> Implement the API in `backend/minuteai/api/`.
>
> Endpoints:
> - `POST /meetings` — create a meeting record, return a presigned S3 upload URL
> - `POST /meetings/{id}/process` — kick off transcription + extraction as a FastAPI BackgroundTask; return immediately with status
> - `GET /meetings/{id}` — meeting metadata + status
> - `GET /meetings/{id}/transcript` — full transcript
> - `GET /meetings/{id}/summary` — the validated `MeetingSummary`
> - `GET /meetings/{id}/dropped` — artifacts the grounding filter rejected (this powers a demo view that makes the novelty visible)
> - `GET /meetings` — list, filterable by series_id and status
> - `GET /meetings/{id}/audio` — presigned playback URL
>
> Pydantic request/response models for everything. Proper status codes. CORS for localhost:3000. A `GET /health` that checks S3 and DynamoDB connectivity.
>
> Status transitions must be persisted at every stage so the frontend can poll meaningfully: uploaded → transcribing → extracting → ready / failed, with the error message stored on failure.
>
> Tests with `TestClient` and moto.

### Prompt 10 — Frontend

> Build the Next.js frontend in `frontend/`. Four routes:
>
> **`/` — meeting list.** Cards showing title, date, duration, participants, status badge. Upload button opening a dialog: drag-drop audio, title, optional series name. Upload goes directly to S3 via the presigned URL, then calls the process endpoint. Poll status every 3s for in-flight meetings.
>
> **`/meetings/[id]` — meeting detail.** The primary view. Three-column layout on desktop, tabs on mobile:
> - Left: transcript, utterance by utterance, with speaker and timestamp. Virtualized — a two-hour meeting is thousands of utterances.
> - Centre: tabs for Summary, Decisions, Action Items.
> - Right: sticky audio player.
>
> **The critical interaction:** hovering any summary sentence, decision, or action item highlights its supporting utterances in the transcript and marks the spans on the audio player's timeline. Clicking it scrolls the transcript to the first supporting utterance and seeks the audio to that timestamp. This single interaction is what makes the grounding contribution visible in a demo — build it properly.
>
> Action items render owner with its confidence. An unresolved owner shows a distinct "Unassigned" state, visually different from an assigned one — never a blank or a guess.
>
> **`/meetings/[id]/dropped` — rejected artifacts.** A table of what the grounding filter removed: artifact text, failure reasons, which checks failed. Framed as "Filtered for insufficient evidence."
>
> **`/search` — natural language search.** Stub the backend call for now; wire it in Phase 9.
>
> Use shadcn/ui components, TanStack Query for data fetching, Zustand only if genuinely needed. Server components where possible, client components where interactivity demands it. Design it to look deliberate — if you need direction, check whether a frontend-design skill is available and use it.

**Verify:** End-to-end — upload a real audio file through the UI, wait for processing, confirm the summary renders and evidence hover works. **You now have a working, demonstrable system.** Tag this commit `v0.1-core`.

---

## Part 8 — Search

### Prompt 11

> Implement search in `backend/minuteai/search/`.
>
> `embedder.py` — sentence-transformers `all-MiniLM-L6-v2` locally (free, fast, good enough). Embeds utterance windows of 5 utterances with stride 2, plus every decision and action item separately.
>
> `index.py` — FAISS `IndexFlatIP` over normalized embeddings. Index persists to S3 at `indexes/global.faiss` with a sidecar JSON mapping vector positions to (meeting_id, utt_range, artifact_type, artifact_id). Rebuild incrementally when a meeting finishes processing.
>
> `hybrid.py` — hybrid retrieval: BM25 over utterance text (use `rank_bm25`) combined with dense retrieval via reciprocal rank fusion. Filters: date range, series, participant, artifact type.
>
> `answerer.py` — given a natural-language query: retrieve top-k, pass retrieved spans to the LLM, and return a `GroundedSentence` answer whose evidence points to real utterances in real meetings. Run the same grounding validator over the answer before returning it. Answers must cite which meeting and which moment.
>
> Endpoint `POST /search` taking query + filters, returning answer plus source list.
>
> Wire the `/search` page: query box, generated answer at top with inline citations, source cards below each linking to the exact moment in the source meeting.

**Verify:** Index three meetings. Ask a question only answerable from one of them. Confirm the answer cites the right meeting and the right moment. Ask a question answerable from none — it must say so rather than fabricating.

---

## Part 9 — Cross-meeting accountability graph *(Novelty 2)*

Only build this if you're committing to the second novelty injection. It's roughly 1.5 weeks.

### Prompt 12

> Implement the cross-meeting accountability graph in `backend/minuteai/graph/`.
>
> `entity_resolution.py` — `resolve_people(meetings) -> dict[str, PersonIdentity]`. Speaker labels are per-meeting and meaningless across meetings ("SPEAKER_01" is a different person each time). Resolve to stable identities using: explicit self-introductions in the transcript, names used in direct address, participant lists supplied at upload, and voice-embedding similarity via pyannote embeddings as a fallback. Every resolution carries a confidence and the method used. Below a threshold, mark unresolved rather than guessing.
>
> `linker.py` — `link_commitments(series_id) -> CommitmentGraph`. For every action item across all meetings in a series, decide whether it is a new commitment or a recurrence of an existing one. Similarity combines: owner identity match, semantic similarity of descriptions (embeddings, threshold ~0.78), artifact/entity overlap, and temporal ordering. Greedy matching against open commitments, newest first.
>
> `lifecycle.py` — each `Commitment` has states:
> ```
> PROPOSED → COMMITTED → { FULFILLED | SLIPPED | SILENTLY_DROPPED | REVERSED }
> ```
> State is inferred from later meetings and every transition carries evidence spans from the meeting that caused it:
> - FULFILLED — a later meeting contains an explicit completion statement referencing it
> - SLIPPED — re-raised in a later meeting with a new deadline or an explicit delay
> - SILENTLY_DROPPED — not mentioned in the subsequent N meetings (configurable, default 2) and past its due date
> - REVERSED — a later decision explicitly contradicts it
>
> `decisions.py` — the same machinery for decisions: detect when a decision in meeting *k* is contradicted by one in meeting *k+n*. Produce a decision-reversal log with both sides' evidence.
>
> Storage: commitments at `PK=SERIES#<id>, SK=COMMITMENT#<id>`, transitions at `PK=COMMITMENT#<id>, SK=TRANSITION#<ts>`.
>
> Endpoints: `GET /series/{id}/commitments`, `GET /series/{id}/reversals`, `GET /people/{id}/ledger`.
>
> Frontend: a `/series/[id]` route showing a commitment timeline (rows = commitments, columns = meetings, cells = state) and a per-person ledger. Every state transition is clickable through to the evidence.

**Verify:** You need a real meeting series — use AMI's ES2002a–d (same team, four sessions) or record 4–5 of your own project meetings. Manually label what actually happened to each commitment, then compare.

---

## Part 10 — Evaluation harness

This is what turns the build into a project. Budget two weeks and do not compress it.

### Prompt 13

> Build the evaluation harness in `eval/`.
>
> `datasets/ami.py` — loader for the AMI Meeting Corpus: audio paths, gold abstractive summaries, gold action-item annotations, gold speaker segmentation. Normalize into our model types.
>
> `datasets/qmsum.py` — loader for QMSum query-summary pairs.
>
> `metrics/summarization.py` — ROUGE-1/2/L and BERTScore against gold summaries.
>
> `metrics/action_items.py` — precision, recall, F1 for action-item detection using semantic matching against gold; owner assignment accuracy; abstention rate; wrong-assignment rate. Report abstention and wrong-assignment separately — conflating them hides the actual contribution.
>
> `metrics/grounding.py` — the headline experiment. Samples N generated sentences stratified across meetings and artifact types, emits them to `eval/results/labelling_sheet.csv` for manual labelling as supported / unsupported / contradicted, then computes hallucination rate from the returned labels. Also computes retention rate and a breakdown of drop reasons from `dropped_artifacts.jsonl`.
>
> `metrics/agreement.py` — Cohen's kappa between two raters on an overlap subset.
>
> `metrics/commitments.py` — lifecycle state classification accuracy and a confusion matrix against manual labels.
>
> `metrics/search.py` — hit@k and MRR on QMSum queries.
>
> `metrics/cost.py` — reads `data/llm_calls.jsonl`; reports tokens, cost, latency per meeting-hour.
>
> `runners/run_ablation.py` — the central experiment. Runs three configurations over the same meeting set:
> - **A**: raw transcript dumped into a single prompt, no chunking, no evidence requirement
> - **B**: chunked + structured extraction with evidence requested, grounding filter OFF
> - **C**: full system, grounding filter ON (run all three strictness modes)
>
> Outputs a comparison table to `eval/results/ablation.csv` and matplotlib charts to `eval/results/figures/`.
>
> `runners/run_all.py` — executes every experiment, writes a consolidated markdown report.
>
> Every runner caches intermediate results so a re-run doesn't re-transcribe or re-extract.

**Verify:** `python eval/runners/run_ablation.py --meetings 5`. The headline number you need: hallucination rate in config B versus config C. If C isn't meaningfully lower, your validator is too permissive — tighten the entity-containment check first.

---

## Part 11 — Deployment and documentation

### Prompt 14

> 1. Dockerize the backend (multi-stage, slim, ffmpeg included). The Whisper model should be baked into the image or downloaded on first boot to a mounted volume.
> 2. `docs/DEPLOYMENT.md` — two paths: local docker-compose, and AWS (ECS Fargate or EC2 for the backend, Amplify or Vercel for the frontend, with the real DynamoDB table and S3 bucket). Include the exact IAM policy JSON the backend needs.
> 3. `scripts/setup_aws.py` — creates the S3 bucket with CORS for presigned uploads, the DynamoDB table with GSIs, and prints the IAM policy to attach.
> 4. `docs/ARCHITECTURE.md` — component diagram in Mermaid, data flow, DynamoDB access-pattern table, the grounding validator's check sequence.
> 5. `README.md` — what it is, screenshots, quickstart, project structure.
> 6. A seed script that loads 3 sample meetings so a fresh clone has something to show.

---

## Part 12 — How to actually run Claude Code

**Use plan mode for every phase.** Press `Shift+Tab` twice to enter plan mode, paste the prompt, read the plan, correct it, *then* approve. The phases above are too large to one-shot.

**`/clear` between phases.** Context from phase 4 actively degrades phase 9. CLAUDE.md carries forward what matters.

**Commit after every phase.** `git commit -m "phase N: <thing>"`. When a phase goes wrong, `git reset --hard` beats arguing with the model about what it broke.

**Keep CLAUDE.md current.** After each phase: *"Update CLAUDE.md to reflect what now exists. Keep it under 150 lines."*

**When something breaks, give evidence not description.** Paste the actual traceback and the actual wrong output. "The grounding validator isn't working" produces guessing; a stack trace plus the dropped-artifacts line that should have passed produces a fix.

**Ask for tests with the code, not after.** Retrofitted tests test what the code does, not what it should do.

**Use `Explore` before changes that touch several modules.** *"Use the Explore agent to find everywhere EvidenceSpan is constructed or consumed."* Then make the change with full knowledge of the blast radius.

**Phases 6, 9, and 10 are the project.** Phases 1–5 and 7 are plumbing you could get from any tutorial. If time runs short, cut frontend polish, cut the AWS Transcribe adapter, cut the deployment phase. Do not cut the grounding validator or the ablation experiment — those are what you defend.

---

## Timeline

| Week | Phases | Output |
|---|---|---|
| 1 | 1–2 | Scaffold, CLAUDE.md, data models, tests green |
| 2 | 3–4 | Storage layer, transcription working on a real file |
| 3 | 5 | LLM extraction producing summaries, decisions, action items |
| 4 | 6 | **Grounding validator + dropped-artifact log** |
| 5 | 7 | API + frontend. **Demoable system. Tag v0.1.** |
| 6 | 8 | Natural language search |
| 7–8 | 9 | Cross-meeting accountability graph *(optional)* |
| 9–10 | 10 | Evaluation: ablation, labelling, all metrics |
| 11 | 11 | Deployment, docs, demo video |
| 12 | — | Report writing |

Drop Phase 9 first if you're behind. The system without it is still a complete project with a defensible contribution.

---

## First three commands

```bash
mkdir minuteai && cd minuteai && git init
claude
```

Then paste Prompt 1. Then Prompt 2. Then commit.
