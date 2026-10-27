.PHONY: install install-ml dev-backend dev-frontend test lint fmt ddb-up ddb-down aws-mock create-table seed sample-audio eval

UV ?= uv

install:
	cd backend && $(UV) sync
	cd frontend && npm install

install-ml:
	cd backend && $(UV) sync --extra ml

dev-backend:
	cd backend && $(UV) run uvicorn minuteai.api.app:app --reload --port 8080

dev-frontend:
	cd frontend && npm run dev

test:
	cd backend && $(UV) run pytest
	cd backend && $(UV) run pytest ../eval/tests

lint:
	cd backend && $(UV) run ruff check minuteai tests ../scripts ../eval
	cd frontend && npm run lint

fmt:
	cd backend && $(UV) run ruff format minuteai tests ../scripts ../eval

ddb-up:
	docker compose up -d dynamodb

ddb-down:
	docker compose down

# In-memory S3 + DynamoDB on :5001 (no Docker). Point both endpoint URLs at it.
aws-mock:
	cd backend && $(UV) run moto_server -H 127.0.0.1 -p 5001

create-table:
	cd backend && $(UV) run python ../scripts/create_table.py

sample-audio:
	cd backend && $(UV) run python ../scripts/make_sample_audio.py

seed:
	cd backend && $(UV) run python ../scripts/seed.py

eval:
	cd backend && $(UV) run python ../eval/runners/run_all.py
