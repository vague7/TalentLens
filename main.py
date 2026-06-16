"""FastAPI Application — Exposes ingestion and screening pipelines.

Provides two main endpoints:
  - POST /ingest: Upload a resume (PDF/DOCX) for preprocessing and vector ingestion.
  - POST /screen: Run the LangGraph multi-agent pipeline given a job description.

Usage:
    uvicorn main:app --reload --port 8000
"""

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_settings
from embeddings import get_embedder
from graph import build_screening_graph
from ingestion import ingest_resume
from logging_config import get_logger
from vectorstore import get_vector_store

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="Resume Screener API",
    description="Multi-agent AI pipeline for fair, structured resume screening.",
    version="1.0.0",
)

# Allow requests from the frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons for ChromaDB and Embeddings
# We use the HTTP mode by default to connect to the docker-compose instance,
# but fallback to persistent if not configured.
vector_mode = "http" if settings.chroma_host else "persistent"
try:
    vector_store = get_vector_store(mode=vector_mode)
    embedder = get_embedder()
    graph = build_screening_graph(embedder=embedder, vector_store=vector_store)
except Exception as e:
    logger.error("api.startup_failed", error=str(e))
    # We don't raise here so the app can start and return 500s dynamically
    vector_store = None
    embedder = None
    graph = None


# ── Request / Response Models ──────────────────────────────────────


class ScreenRequest(BaseModel):
    """Payload for the /screen endpoint."""
    jd_text: str
    top_k: int = 5


# ── Endpoints ──────────────────────────────────────────────────────


@app.get("/health")
def health_check() -> dict[str, str]:
    """Check API and Vector Store health."""
    status = "healthy"
    try:
        if vector_store:
            vector_store.count()
    except Exception:
        status = "degraded"
        
    return {"status": status}


@app.post("/ingest")
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_ner: bool = Form(True),
) -> dict[str, Any]:
    """Upload a resume file and ingest it into the vector store.
    
    The file is saved to a temp directory, processed through the pipeline
    (parse -> anonymize -> chunk -> embed -> store), and then deleted.
    """
    if not vector_store or not embedder:
        raise HTTPException(500, "Vector store or embedder not initialized")

    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".docx"]:
        raise HTTPException(400, "Only .pdf and .docx files are supported")

    # Save uploaded file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run ingestion pipeline
        result = ingest_resume(
            file_path=temp_path,
            embedder=embedder,
            vector_store=vector_store,
            use_ner=use_ner,
        )
        return {
            "candidate_id": result.candidate_id,
            "chunks_stored": result.chunks_stored,
            "sections_detected": result.sections,
            "anonymization_count": result.anonymization_count,
            "processing_time_ms": result.processing_time_ms,
            "status": "success",
        }
    except Exception as e:
        logger.error("api.ingest_failed", filename=file.filename, error=str(e))
        raise HTTPException(500, f"Ingestion failed: {e}")
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


@app.post("/screen")
async def screen_endpoint(request: ScreenRequest) -> dict[str, Any]:
    """Run the multi-agent screening pipeline.
    
    Takes a Job Description and evaluates all ingested candidates
    using the LangGraph pipeline (screen -> score -> audit).
    """
    if not graph:
        raise HTTPException(500, "LangGraph pipeline not initialized")

    if not request.jd_text.strip():
        raise HTTPException(400, "Job description cannot be empty")

    logger.info("api.screen_request", top_k=request.top_k)

    try:
        # Invoke the LangGraph pipeline
        initial_state = {
            "jd_text": request.jd_text,
            "top_k": request.top_k,
        }
        
        final_state = graph.invoke(initial_state)

        # Build response
        error = final_state.get("error")
        if error:
            raise HTTPException(500, f"Pipeline error: {error}")

        bias_report = final_state.get("bias_report", {})
        release_blocked = final_state.get("release_blocked", False)
        
        # If release is blocked, we don't return scores
        if release_blocked:
            return {
                "status": "human_review_required",
                "message": "Results blocked by Bias Auditor due to high risk.",
                "bias_report": bias_report,
                "scores": [],
            }

        return {
            "status": "success",
            "message": "Screening complete.",
            "bias_report": bias_report,
            "scores": final_state.get("scores", []),
            "jd_criteria": final_state.get("jd_criteria", []),
        }

    except Exception as e:
        logger.error("api.screen_failed", error=str(e))
        raise HTTPException(500, f"Screening pipeline failed: {e}")
