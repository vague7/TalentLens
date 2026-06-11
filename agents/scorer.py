"""Agent 2 — Structured LLM Scorer.

Scores shortlisted candidates against JD criteria using GPT-4o with
structured output (response_format = pydantic model). Every criterion
score must include a verbatim quote from the resume chunk.

Design:
  - For each shortlisted candidate, retrieves their chunks from ChromaDB
  - Sends chunks + JD criteria to GPT-4o with structured output
  - Returns CandidateScore with per-criterion scores and evidence
  - model_validator enforces evidence-in-reasoning invariant
"""

from __future__ import annotations

import json
import time
from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from logging_config import get_logger
from vectorstore import VectorStore

logger = get_logger(__name__)


# ── Pydantic Output Models ─────────────────────────────────────────


class CriterionScore(BaseModel):
    """Score for a single evaluation criterion.

    Attributes:
        criterion: The JD criterion being evaluated.
        score: Integer score from 1 to 10.
        reasoning: Explanation that MUST contain the verbatim evidence quote.
        evidence: The exact quote from the resume chunk.
    """

    criterion: str
    score: int = Field(ge=1, le=10)
    reasoning: str  # must contain a verbatim quote
    evidence: str  # the exact quote from the resume chunk

    @model_validator(mode="after")
    def evidence_in_reasoning(self) -> "CriterionScore":
        """Validate that the evidence quote appears verbatim in the reasoning."""
        if self.evidence not in self.reasoning:
            raise ValueError(
                "reasoning must contain the evidence quote verbatim. "
                f"Evidence: '{self.evidence}' not found in reasoning."
            )
        return self


class CandidateScore(BaseModel):
    """Aggregated score for a single candidate.

    Attributes:
        candidate_id: Anonymized UUID for the candidate.
        overall_score: Weighted aggregate score (0.0–10.0).
        criteria: List of per-criterion scores.
        recommendation: Final recommendation action.
        confidence: Model confidence in the assessment (0.0–1.0).
        processing_time_ms: Wall-clock time for scoring this candidate.
    """

    candidate_id: str
    overall_score: float = Field(ge=0.0, le=10.0)
    criteria: list[CriterionScore]
    recommendation: Literal["shortlist", "hold", "reject"]
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time_ms: int  # log this for latency monitoring


# ── Internal LLM response model (for structured output) ───────────


class ScorerLLMResponse(BaseModel):
    """Internal model for the LLM structured output.

    This is what GPT-4o returns. We then wrap it into CandidateScore
    with additional metadata (candidate_id, processing_time_ms).
    """

    overall_score: float = Field(ge=0.0, le=10.0)
    criteria: list[CriterionScore]
    recommendation: Literal["shortlist", "hold", "reject"]
    confidence: float = Field(ge=0.0, le=1.0)


# ── Scorer Agent ───────────────────────────────────────────────────


SCORER_SYSTEM_PROMPT = """You are an expert technical recruiter scoring a candidate's resume \
against specific job criteria.

For each criterion, you must:
1. Find the most relevant evidence in the resume chunks provided.
2. Quote the evidence VERBATIM from the resume — do not paraphrase.
3. Include that exact verbatim quote in your reasoning explanation.
4. Assign a score from 1 (no match) to 10 (perfect match).

Rules:
- The "evidence" field must be an EXACT substring copied from the resume text.
- The "reasoning" field must contain the evidence quote within it.
- If no relevant evidence exists for a criterion, use evidence="" and score 1-2.
- Be objective and consistent across all criteria.
- Overall score should be the weighted average of criterion scores.
- Recommendation: "shortlist" if overall >= 7, "hold" if 5-6.9, "reject" if < 5.
- Confidence reflects how much resume evidence was available (0.0 = no evidence, 1.0 = strong evidence for all criteria).

Return a JSON object with: overall_score, criteria (array), recommendation, confidence."""


def score_candidate(
    candidate_id: str,
    jd_criteria: list[str],
    resume_chunks: list[str],
    llm_client: OpenAI,
    model: str = "gpt-4o",
) -> CandidateScore:
    """Score a single candidate against JD criteria using GPT-4o.

    Args:
        candidate_id: Anonymized UUID of the candidate.
        jd_criteria: List of criterion strings to evaluate against.
        resume_chunks: List of resume chunk texts for this candidate.
        llm_client: Configured OpenAI client.
        model: LLM model to use.

    Returns:
        CandidateScore with per-criterion scores and evidence.
    """
    start_time = time.perf_counter()

    # Prepare the user prompt with resume chunks and criteria
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(resume_chunks)
    )

    criteria_text = "\n".join(f"- {c}" for c in jd_criteria)

    user_prompt = f"""## Resume Chunks

{chunks_text}

## Evaluation Criteria

{criteria_text}

Score this candidate on each criterion. For each, find a verbatim quote from the resume chunks above as evidence."""

    # Call LLM with structured output
    llm_response = _call_scorer_llm(
        llm_client=llm_client,
        model=model,
        system_prompt=SCORER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    # Wrap into CandidateScore with metadata
    result = CandidateScore(
        candidate_id=candidate_id,
        overall_score=llm_response.overall_score,
        criteria=llm_response.criteria,
        recommendation=llm_response.recommendation,
        confidence=llm_response.confidence,
        processing_time_ms=elapsed_ms,
    )

    logger.info(
        "scorer.candidate_scored",
        candidate_id=candidate_id,
        overall_score=result.overall_score,
        recommendation=result.recommendation,
        confidence=result.confidence,
        criteria_count=len(result.criteria),
        processing_time_ms=elapsed_ms,
    )

    return result


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _call_scorer_llm(
    llm_client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> ScorerLLMResponse:
    """Call GPT-4o with structured output for scoring.

    Args:
        llm_client: OpenAI client instance.
        model: Model name.
        system_prompt: System prompt for the scorer.
        user_prompt: User prompt with resume chunks and criteria.

    Returns:
        Parsed ScorerLLMResponse.
    """
    start_time = time.perf_counter()

    response = llm_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    logger.info("scorer.llm_call", model=model, response_time_ms=elapsed_ms)

    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)

    return ScorerLLMResponse(**parsed)


def score_all_candidates(
    shortlisted_ids: list[str],
    jd_criteria: list[str],
    vector_store: VectorStore,
    llm_client: OpenAI,
    model: str = "gpt-4o",
) -> list[CandidateScore]:
    """Score all shortlisted candidates against JD criteria.

    Args:
        shortlisted_ids: List of candidate UUIDs to score.
        jd_criteria: Evaluation criteria extracted from the JD.
        vector_store: Vector store to retrieve candidate chunks.
        llm_client: OpenAI client for LLM calls.
        model: LLM model to use.

    Returns:
        List of CandidateScore objects, one per candidate.
    """
    logger.info(
        "scorer.batch_start",
        candidates=len(shortlisted_ids),
        criteria=len(jd_criteria),
    )

    scores: list[CandidateScore] = []

    for cid in shortlisted_ids:
        # Retrieve chunks for this candidate from ChromaDB
        chunks_data = vector_store.get_chunks_by_candidate(cid)
        chunk_texts = [c["text"] for c in chunks_data if c.get("text")]

        if not chunk_texts:
            logger.warning("scorer.no_chunks", candidate_id=cid)
            continue

        try:
            score = score_candidate(
                candidate_id=cid,
                jd_criteria=jd_criteria,
                resume_chunks=chunk_texts,
                llm_client=llm_client,
                model=model,
            )
            scores.append(score)
        except Exception as e:
            logger.error(
                "scorer.candidate_failed",
                candidate_id=cid,
                error=str(e),
            )

    logger.info(
        "scorer.batch_complete",
        scored=len(scores),
        total=len(shortlisted_ids),
    )

    return scores
