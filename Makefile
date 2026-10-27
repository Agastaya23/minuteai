.PHONY: install dev-backend dev-frontend test lint fmt ddb-up ddb-down create-table seed eval

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
	cd eval && ../backend/.venv/bin/python -m pytest -q

lint:
	cd backend && $(UV) run ruff check minuteai tests ../scripts ../eval
	cd frontend && npm run lint

fmt:
	cd backend && $(UV) run ruff format minuteai tests ../scripts ../eval

ddb-up:
	docker compose up -d dynamodb

ddb-down:
	docker compose down

create-table:
	cd backend && $(UV) run python ../scripts/create_table.py

seed:
	cd backend && $(UV) run python ../scripts/seed.py

eval:
	cd backend && $(UV) run python ../eval/runners/run_all.py
