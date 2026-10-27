# Deployment

Two paths: everything local with docker compose, or AWS (backend on ECS Fargate or EC2, frontend on Vercel or Amplify, real DynamoDB and S3).

## Configuration

All backend settings come from environment variables (see `.env.example`):

| Variable | Notes |
|---|---|
| `AWS_REGION`, `S3_BUCKET`, `DDB_TABLE` | where the data lives |
| `DDB_ENDPOINT_URL` | empty for real DynamoDB; `http://dynamodb:8000` in compose |
| `S3_ENDPOINT_URL` | empty for real S3 (only set for local mocks) |
| `LLM_MODEL`, `LLM_API_KEY` | `claude-*` uses Anthropic and `gemini-*` uses Google. The key can also come from `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`. |
| `HF_TOKEN` | pyannote diarization. Accept the licences for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` first. |
| `WHISPER_MODEL` | `small` by default; `base` is faster, `medium` more accurate |
| `GROUNDING_MODE` | `STRICT` \| `BALANCED` \| `PERMISSIVE` |
| `CORS_ORIGINS` | JSON list of frontend origins, e.g. `["https://minuteai.example.com"]` |

The frontend needs only `NEXT_PUBLIC_API_URL`, which is baked in at build time.

## Path 1 — local, docker compose

```bash
cp .env.example .env            # fill in AWS creds, S3_BUCKET, LLM_API_KEY, HF_TOKEN
uv --directory backend run python ../scripts/setup_aws.py --print-policy   # attach to your IAM user
docker compose --profile app up --build
# UI http://localhost:3000 · API http://localhost:8080/docs · DynamoDB Local :8000
docker compose exec backend python -c "from minuteai.storage.schema import create_table; create_table()"
```

Whisper and pyannote models download on first use into the `models` volume, so the first transcription is slow and later ones aren't. To bake the diarization and retrieval stack into the image, build with `--build-arg EXTRAS="ml"`. The image gets much larger because of torch.

No Docker? `make aws-mock` runs moto as an in-memory S3 + DynamoDB on :5001. Point `DDB_ENDPOINT_URL` and `S3_ENDPOINT_URL` at it, then `make seed && make dev-backend && make dev-frontend`.

## Path 2 — AWS

### 1. Provision data stores

```bash
cd backend
uv run python ../scripts/setup_aws.py --bucket minuteai-prod --table minuteai \
    --region ap-south-1 --origin https://minuteai.example.com
```

This creates the bucket (public access blocked, CORS for presigned PUT/GET from your frontend origin) and the table with GSI1 (on-demand), then prints the IAM policy for the backend.

### 2. IAM policy for the backend role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "MeetingAudioTranscriptsIndexes",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::minuteai-prod/*"
    },
    {
      "Sid": "BucketLevel",
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": "arn:aws:s3:::minuteai-prod"
    },
    {
      "Sid": "SingleTable",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem",
        "dynamodb:Query", "dynamodb:BatchWriteItem", "dynamodb:BatchGetItem",
        "dynamodb:DescribeTable"
      ],
      "Resource": [
        "arn:aws:dynamodb:ap-south-1:<ACCOUNT_ID>:table/minuteai",
        "arn:aws:dynamodb:ap-south-1:<ACCOUNT_ID>:table/minuteai/index/*"
      ]
    },
    {
      "Sid": "OptionalAwsTranscribeBackend",
      "Effect": "Allow",
      "Action": ["transcribe:StartTranscriptionJob", "transcribe:GetTranscriptionJob"],
      "Resource": "*"
    }
  ]
}
```

Presigned URLs are signed by the backend role, so the browser needs no AWS credentials.

### 3. Backend — ECS Fargate (recommended)

```bash
aws ecr create-repository --repository-name minuteai-backend
docker build -t minuteai-backend backend/
docker tag minuteai-backend <acct>.dkr.ecr.<region>.amazonaws.com/minuteai-backend:latest
docker push <acct>.dkr.ecr.<region>.amazonaws.com/minuteai-backend:latest
```

* Task: 2 vCPU / 8 GB is comfortable for Whisper `small` on CPU. Mount EFS at `/models` so model downloads persist across tasks.
* Task role: the policy above. Pass `LLM_API_KEY` / `HF_TOKEN` from Secrets Manager.
* Put it behind an ALB with health check `GET /health` on port 8080.
* Processing runs as a FastAPI background task inside the API container, so run **one task**, or move `process_meeting` to a queue worker (SQS + a second service) before scaling out.

**EC2 alternative:** a `g4dn.xlarge` with the Docker image and `--gpus all` (set `device="cuda"`) cuts transcription time roughly 10×.

### 4. Frontend — Vercel or Amplify

* **Vercel:** import the repo, set the root directory to `frontend/`, and set `NEXT_PUBLIC_API_URL=https://api.minuteai.example.com`.
* **Amplify Hosting:** connect the repo, set the app root to `frontend`, and use the same env var. Next.js SSR is supported.
* Add the frontend origin to `CORS_ORIGINS` on the backend *and* to the S3 bucket CORS (`setup_aws.py --origin …`).

### 5. Seed and smoke-test

```bash
curl https://api.minuteai.example.com/health        # {"status":"ok","s3":"ok","dynamodb":"ok",...}
uv --directory backend run python ../scripts/seed.py # optional demo data
```
