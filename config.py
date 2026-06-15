"""Resume Screener — Centralized application settings.

All configuration is loaded from environment variables (with .env file support).
No hardcoded secrets — every sensitive value comes from os.getenv().
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI ──────────────────────────────
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o"

    # ── ChromaDB ────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "resumes"

    # ── FastAPI ─────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_log_level: str = "info"

    # ── Evaluation thresholds ───────────────
    eval_faithfulness_threshold: float = 0.80
    eval_answer_relevancy_threshold: float = 0.75
    eval_context_precision_threshold: float = 0.70

    # ── Paths ───────────────────────────────
    upload_dir: Path = Path("./data/uploads")
    results_dir: Path = Path("./results")

    # ── spaCy ───────────────────────────────
    spacy_model: str = "en_core_web_trf"


def get_settings() -> Settings:
    """Return a cached Settings instance.

    Returns:
        Settings: The application settings singleton.
    """
    return Settings()
