# syntax=docker/dockerfile:1.7
# MinuteAI backend. Multi-stage: build the venv with uv, ship a slim runtime with ffmpeg.
#   docker build -t minuteai-backend backend/                         # API + Whisper
#   docker build -t minuteai-backend --build-arg EXTRAS="ml" backend/     # + pyannote/faiss/st/spacy
# Whisper / HF models are downloaded on first use into /models (mount a volume there).

FROM python:3.12-slim AS build
ARG EXTRAS="asr"
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
      $(for e in $EXTRAS; do printf -- "--extra %s " "$e"; done)
COPY minuteai ./minuteai
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev \
      $(for e in $EXTRAS; do printf -- "--extra %s " "$e"; done)

FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app && mkdir -p /models /data && chown app /models /data
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    XDG_CACHE_HOME=/models \
    DATA_DIR=/data
WORKDIR /app
COPY --from=build /app /app
USER app
VOLUME ["/models", "/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"
CMD ["uvicorn", "minuteai.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
