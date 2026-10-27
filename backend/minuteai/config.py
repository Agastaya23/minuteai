"""Runtime configuration, read once from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(_REPO_ROOT / ".env"), ".env"), extra="ignore"
    )

    aws_region: str = "ap-south-1"
    s3_bucket: str = "minuteai-dev"
    ddb_table: str = "minuteai"
    ddb_endpoint_url: str | None = None
    s3_endpoint_url: str | None = None

    llm_api_key: str | None = None
    llm_model: str = "claude-haiku-5-5"
    # Reasoning depth for providers that think by default (Gemini 3.x).
    # Extraction is mostly copying with structure; "low" is ~10x faster than the default.
    llm_thinking: str = "low"

    hf_token: str | None = None
    whisper_model: str = "small"
    transcribe_backend: str = "whisper"

    grounding_mode: str = "BALANCED"
    data_dir: Path = _REPO_ROOT / "data"
    cors_origins: list[str] = ["http://localhost:3000"]

    def data_path(self, name: str) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / name


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ddb_endpoint_url = settings.ddb_endpoint_url or None
    settings.s3_endpoint_url = settings.s3_endpoint_url or None
    return settings
