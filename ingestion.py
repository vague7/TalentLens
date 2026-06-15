"""Resume ingestion pipeline — parse → anonymize → chunk → embed → store.

Orchestrates the full preprocessing and vector store ingestion for a
single resume document. This is the main entry point for the /ingest
API endpoint.

Usage:
    result = await ingest_resume(file_path, embedder, vector_store)
    print(result.candidate_id, result.chunks_stored)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from embeddings import ResumeEmbedder
from logging_config import get_logger
from preprocessing.anonymizer import AnonymizationResult, anonymize_text
from preprocessing.chunker import ChunkResult, ResumeChunk, chunk_resume
from preprocessing.parser import ParseResult, parse_document
from vectorstore import VectorStore

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Result of ingesting a single resume document.

    Attributes:
        candidate_id: UUID assigned to this candidate.
        file_name: Original file name.
        chunks_stored: Number of chunks stored in the vector store.
        total_sections: Number of sections detected in the resume.
        sections: List of section names found.
        anonymization_count: Number of PII replacements made.
        processing_time_ms: Total end-to-end processing time.
    """

    candidate_id: str
    file_name: str
    chunks_stored: int
    total_sections: int
    sections: list[str]
    anonymization_count: int
    processing_time_ms: int


def ingest_resume(
    file_path: Path,
    embedder: ResumeEmbedder,
    vector_store: VectorStore,
    candidate_uuid: Optional[str] = None,
    use_ner: bool = True,
    max_chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> IngestionResult:
    """Run the full ingestion pipeline for a single resume.

    Pipeline stages:
        1. Parse: Extract text from PDF/DOCX
        2. Anonymize: Strip PII (names, emails, phones, etc.)
        3. Chunk: Split into section-aware chunks with metadata
        4. Embed: Generate OpenAI embeddings for each chunk
        5. Store: Upsert chunks + embeddings into ChromaDB

    Args:
        file_path: Path to the resume file (PDF or DOCX).
        embedder: Configured ResumeEmbedder instance.
        vector_store: Configured VectorStore instance.
        candidate_uuid: Optional pre-assigned UUID. Auto-generated if None.
        use_ner: Whether to use spaCy NER for name anonymization.
        max_chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        IngestionResult with processing details.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is unsupported or contains no text.
    """
    start_time = time.perf_counter()
    cid = candidate_uuid or str(uuid.uuid4())

    logger.info("ingestion.start", candidate_id=cid, file=str(file_path))

    # ── Stage 1: Parse ──────────────────────────────────────────
    parse_result: ParseResult = parse_document(file_path)
    logger.info(
        "ingestion.parsed",
        candidate_id=cid,
        pages=parse_result.total_pages,
        chars=len(parse_result.full_text),
    )

    # ── Stage 2: Anonymize ──────────────────────────────────────
    # CRITICAL: Anonymization MUST happen before any agent sees the text
    anon_result: AnonymizationResult = anonymize_text(
        text=parse_result.full_text,
        candidate_uuid=cid,
        use_ner=use_ner,
    )
    logger.info(
        "ingestion.anonymized",
        candidate_id=cid,
        replacements=anon_result.replacement_count,
        categories=list(anon_result.replacements.keys()),
    )

    # ── Stage 3: Chunk ──────────────────────────────────────────
    chunk_result: ChunkResult = chunk_resume(
        text=anon_result.anonymized_text,
        candidate_id=cid,
        max_chunk_size=max_chunk_size,
        overlap=chunk_overlap,
    )
    logger.info(
        "ingestion.chunked",
        candidate_id=cid,
        chunks=chunk_result.total_chunks,
        sections=chunk_result.sections_detected,
    )

    if chunk_result.total_chunks == 0:
        logger.warning("ingestion.no_chunks", candidate_id=cid)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        return IngestionResult(
            candidate_id=cid,
            file_name=parse_result.file_name,
            chunks_stored=0,
            total_sections=0,
            sections=[],
            anonymization_count=anon_result.replacement_count,
            processing_time_ms=elapsed_ms,
        )

    # ── Stage 4: Embed ──────────────────────────────────────────
    chunk_texts = [chunk.text for chunk in chunk_result.chunks]
    embeddings = embedder.embed_texts(chunk_texts)

    # ── Stage 5: Store ──────────────────────────────────────────
    chunk_ids = [f"{cid}_chunk_{chunk.chunk_index}" for chunk in chunk_result.chunks]
    metadatas = [
        {
            "candidate_id": cid,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "char_count": chunk.char_count,
            "file_name": parse_result.file_name,
            **{k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))},
        }
        for chunk in chunk_result.chunks
    ]

    stored_count = vector_store.add_chunks(
        chunk_ids=chunk_ids,
        texts=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    logger.info(
        "ingestion.complete",
        candidate_id=cid,
        chunks_stored=stored_count,
        processing_time_ms=elapsed_ms,
    )

    return IngestionResult(
        candidate_id=cid,
        file_name=parse_result.file_name,
        chunks_stored=stored_count,
        total_sections=len(chunk_result.sections_detected),
        sections=chunk_result.sections_detected,
        anonymization_count=anon_result.replacement_count,
        processing_time_ms=elapsed_ms,
    )
