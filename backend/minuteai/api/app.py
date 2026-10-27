"""FastAPI application factory."""

from __future__ import annotations

import logging

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from minuteai.api.processing import Services
from minuteai.api.routes_meetings import router as meetings_router
from minuteai.api.schemas import HealthResponse
from minuteai.config import get_settings
from minuteai.storage.repository import MeetingRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def default_services() -> Services:
    from minuteai.api.hooks import default_hooks
    from minuteai.graph.service import GraphService
    from minuteai.search.service import SearchService

    repo = MeetingRepository()
    services = Services(repo=repo, search=SearchService(repo), graph=GraphService(repo))

    services.on_ready.extend(default_hooks(services))
    return services


def create_app(services: Services | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="MinuteAI", version="0.1.0",
                  description="Grounded meeting intelligence API")
    app.state.services = services or default_services()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(meetings_router)
    from minuteai.api.routes_graph import router as graph_router
    from minuteai.api.routes_search import router as search_router

    app.include_router(search_router)
    app.include_router(graph_router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request) -> HealthResponse:
        repo: MeetingRepository = request.app.state.services.repo
        checks = {}
        for name, probe in (("s3", repo.s3.ping), ("dynamodb", repo.ping)):
            try:
                probe()
                checks[name] = "ok"
            except (ClientError, BotoCoreError) as e:
                checks[name] = f"error: {e}"[:200]
        ok = all(v == "ok" for v in checks.values())
        return HealthResponse(status="ok" if ok else "degraded", llm_model=settings.llm_model,
                              **checks)

    return app


def _lazy_app() -> FastAPI:
    return create_app()


app = _lazy_app()
