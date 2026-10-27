# MinuteAI

Meeting intelligence platform. Audio is transcribed with word-level timestamps and speaker
diarization; an LLM extracts summaries, decisions and action items; a deterministic grounding
validator checks every artifact against the transcript and drops (and logs) what it cannot
support. Novelties: (1) the grounding validator, (2) the cross-meeting accountability graph.

## Pipeline

```
audio ─▶ S3 ─▶ transcribe (faster-whisper + pyannote │ AWS Transcribe)
           ─▶ Transcript[Utterance{utt_id, speaker, start, end, words[]}]
           ─▶ chunk ─▶ LLM extraction (evidence: utt_id + quote per claim)
           ─▶ GroundingValidator / GroundingFilter ─▶ kept │ dropped_artifacts.jsonl
           ─▶ DynamoDB (single table) + S3 ─▶ search index (FAISS/BM25) ─▶ FastAPI ─▶ Next.js
```

## Hard rules (non-negotiable)

- Every extracted artifact (summary sentence, decision, action item, search answer) MUST carry
  `evidence: list[EvidenceSpan]` pointing to real utterance IDs and timestamps. No exceptions.
- The LLM never determines structure, only content. Structure (ids, timestamps, speaker labels,
  chunking, merging, dedup) comes from deterministic Python.
- All LLM calls go through `minuteai/extract/llm_client.py`. No direct SDK calls anywhere else.
- All LLM responses are parsed into pydantic models. Never consume raw text.
- All storage access goes through `minuteai/storage/`. No boto3 calls elsewhere.
- Transcription backends are adapters behind `TranscriptionBackend`. No backend-specific types
  leak out.
- Action-item owners are nullable: `owner=None` + `owner_resolution_method="unresolved"` beats
  a guess.

## Commands

| Task | Command |
|---|---|
| Install | `make install` (backend via uv, frontend via npm); `make install-ml` for whisper/pyannote/faiss |
| Backend dev server (:8080) | `make dev-backend` |
| Frontend dev server (:3000) | `make dev-frontend` |
| Tests | `make test` (or `cd backend && uv run pytest tests/test_x.py -k name`) |
| Lint | `make lint` |
| DynamoDB Local (:8000) | `make ddb-up` / `make ddb-down`, then `make create-table` |
| Seed 3 demo meetings | `make seed` |
| Transcribe a file | `cd backend && uv run python ../scripts/transcribe_local.py path.wav` |
| Extract from cached transcript | `cd backend && uv run python ../scripts/extract_local.py <meeting_id>` |
| Evaluation | `make eval`; ablation: `uv run python ../eval/runners/run_ablation.py --meetings 5` |

Python 3.12 (managed by uv; avoid 3.13+ for pyannote). Node 20+.

## Conventions

- pydantic v2 models for all data crossing a module boundary; type hints everywhere.
- pytest; shared fixtures (`transcript`, `span`, `aws`) in `backend/tests/conftest.py`.
  AWS is mocked with moto, the LLM with a fake provider — tests never hit the network.
- No bare `except`. Catch specific exceptions.
- Heavy ML deps (faster-whisper, pyannote, sentence-transformers, faiss, spacy) are optional
  and imported lazily; every module has a lightweight fallback so tests run without them.
- Settings come from `minuteai.config.get_settings()` (env / `.env`), never `os.environ` directly.
- Research logs: `data/llm_calls.jsonl` (every LLM call) and `data/dropped_artifacts.jsonl`
  (every grounding reject). Don't remove fields from them — the eval harness reads them.

## Where things live

| Path | Responsibility |
|---|---|
| `backend/minuteai/models/` | Contract: `transcript`, `evidence`, `artifacts`, `meeting`, `grounding`, `graph` |
| `backend/minuteai/storage/` | `schema.py` (key design + table creation), `repository.py`, `s3_store.py` |
| `backend/minuteai/transcribe/` | `base.py` interface, whisper/aws backends, `diarization.py`, `pipeline.py` |
| `backend/minuteai/extract/` | `llm_client.py` (only LLM entry), `chunker.py`, `prompts.py`, `extractor.py` |
| `backend/minuteai/ground/` | `validator.py` (6 checks), `filter.py` (STRICT/BALANCED/PERMISSIVE) |
| `backend/minuteai/search/` | `embedder.py`, `index.py` (FAISS→S3), `hybrid.py` (BM25+dense RRF), `answerer.py` |
| `backend/minuteai/graph/` | `entity_resolution.py`, `linker.py`, `lifecycle.py`, `decisions.py`, `service.py` |
| `backend/minuteai/api/` | FastAPI app, routes, background processing |
| `backend/minuteai/samples.py` | Synthetic 3-meeting series used by tests, seed, offline eval |
| `frontend/` | Next.js 15 App Router UI (`/`, `/meetings/[id]`, `/meetings/[id]/dropped`, `/search`, `/series/[id]`) |
| `eval/` | Datasets (AMI, QMSum), metrics, runners (`run_ablation.py`, `run_all.py`) |
| `scripts/` | `create_table.py`, `setup_aws.py`, `transcribe_local.py`, `extract_local.py`, `seed.py` |
| `docs/` | `ARCHITECTURE.md`, `DEPLOYMENT.md`, `BUILD_GUIDE.md` |
