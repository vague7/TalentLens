"""Agent 1 — RAG-based Resume Screener.

Retrieves top-K candidate resumes from ChromaDB using semantic similarity
against the job description. Performs initial shortlisting before scoring.

Design:
  - Embeds the full JD text as the query vector
  - Queries ChromaDB for top-K most similar chunks
  - Aggregates per-candidate similarity (mean of top chunk scores)
  - Returns shortlisted candidate IDs ranked by aggregate similarity
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from embeddings import ResumeEmbedder
from logging_config import get_logger
from vectorstore import VectorStore

logger = get_logger(__name__)


@dataclass
class CandidateMatch:
    """A candidate's aggregated match result from screening.

    Attributes:
        candidate_id: UUID of the candidate.
        avg_similarity: Mean similarity score across matched chunks.
        top_chunks: Best matching chunk texts for this candidate.
        chunk_count: Number of chunks that matched.
    """

    candidate_id: str
    avg_similarity: float
    top_chunks: list[str] = field(default_factory=list)
    chunk_count: int = 0


@dataclass
class ScreeningResult:
    """Result of the screening agent.

    Attributes:
        shortlisted: Ranked list of candidate matches.
        total_candidates: Number of unique candidates considered.
        processing_time_ms: Wall-clock time for screening.
    """

    shortlisted: list[CandidateMatch] = field(default_factory=list)
    total_candidates: int = 0
    processing_time_ms: int = 0


def screen_candidates(
    jd_text: str,
    embedder: ResumeEmbedder,
    vector_store: VectorStore,
    top_k: int = 5,
    chunks_per_query: int = 50,
) -> ScreeningResult:
    """Screen candidates by semantic similarity to the job description.

    Embeds the JD, queries ChromaDB for similar chunks, aggregates
    scores per candidate, and returns the top-K shortlist.

    Args:
        jd_text: Full job description text.
        embedder: Configured ResumeEmbedder for query embedding.
        vector_store: ChromaDB vector store with ingested resumes.
        top_k: Number of candidates to shortlist.
        chunks_per_query: Number of chunks to retrieve from vector store.

    Returns:
        ScreeningResult with ranked shortlisted candidates.
    """
    start_time = time.perf_counter()

    logger.info("screener.start", top_k=top_k, jd_length=len(jd_text))

    # Step 1: Embed the job description
    jd_embedding = embedder.embed_query(jd_text)

    # Step 2: Query vector store for similar chunks
    query_result = vector_store.query(
        query_embedding=jd_embedding,
        top_k=chunks_per_query,
    )

    # Step 3: Aggregate similarity per candidate
    # ChromaDB returns distances (lower = more similar for cosine)
    # Convert to similarity: 1 - distance
    candidate_scores: dict[str, list[float]] = defaultdict(list)
    candidate_chunks: dict[str, list[str]] = defaultdict(list)

    for result in query_result.results:
        cid = result.candidate_id
        similarity = 1.0 - result.distance  # Convert distance to similarity
        candidate_scores[cid].append(similarity)
        candidate_chunks[cid].append(result.text)

    # Step 4: Compute average similarity and rank
    matches: list[CandidateMatch] = []
    for cid, scores in candidate_scores.items():
        avg_sim = sum(scores) / len(scores)
        # Keep top 3 chunks per candidate for context
        top_texts = candidate_chunks[cid][:3]
        matches.append(
            CandidateMatch(
                candidate_id=cid,
                avg_similarity=round(avg_sim, 4),
                top_chunks=top_texts,
                chunk_count=len(scores),
            )
        )

    # Sort by similarity (descending) and take top-K
    matches.sort(key=lambda m: m.avg_similarity, reverse=True)
    shortlisted = matches[:top_k]

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    logger.info(
        "screener.complete",
        total_candidates=len(matches),
        shortlisted=len(shortlisted),
        top_score=shortlisted[0].avg_similarity if shortlisted else 0,
        processing_time_ms=elapsed_ms,
    )

    return ScreeningResult(
        shortlisted=shortlisted,
        total_candidates=len(matches),
        processing_time_ms=elapsed_ms,
    )


def extract_jd_criteria(jd_text: str, llm_client: Any) -> list[str]:
    """Extract evaluation criteria from a job description using LLM.

    Parses the JD to identify specific, scoreable requirements
    that the scorer agent will evaluate candidates against.

    Args:
        jd_text: Full job description text.
        llm_client: Configured OpenAI client for LLM calls.

    Returns:
        List of criterion strings extracted from the JD.
    """
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_llm() -> list[str]:
        start_time = time.perf_counter()

        response = llm_client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert HR analyst. Extract the key evaluation "
                        "criteria from the following job description. Return them as "
                        "a JSON array of strings, each being a specific, scoreable "
                        "requirement. Include technical skills, experience levels, "
                        "soft skills, and qualifications mentioned. Return 5-10 criteria."
                    ),
                },
                {"role": "user", "content": jd_text},
            ],
            response_format={"type": "json_object"},
        )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("screener.extract_criteria", response_time_ms=elapsed_ms)

        import json
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        # Handle both {"criteria": [...]} and direct array
        if isinstance(parsed, list):
            return parsed
        return parsed.get("criteria", parsed.get("requirements", []))

    return _call_llm()
