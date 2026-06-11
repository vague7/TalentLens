"""FastAPI application — Resume Screener REST API.

Endpoints:
  POST /ingest          Upload resume file → returns {candidate_id}
  POST /screen          Body: {jd_text, top_k} → runs full pipeline
  GET  /status/{id}     Returns pipeline stage + partial results
  GET  /results/{run}   Returns full CandidateScore list + BiasReport
  GET  /health          Liveness check
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from config import get_settings
from logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown hooks.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    settings = get_settings()
    setup_logging(settings.api_log_level)
    logger.info("resume_screener.startup", host=settings.api_host, port=settings.api_port)

    # Ensure required directories exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)

    yield

    logger.info("resume_screener.shutdown")


app = FastAPI(
    title="Resume Screener API",
    description="Multi-agent AI Resume Screener with RAG, structured scoring, and bias auditing.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Liveness check endpoint.

    Returns:
        A dict with status "ok".
    """
    return {"status": "ok"}


# Full endpoint implementations in Phase 6
