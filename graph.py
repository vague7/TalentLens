"""LangGraph pipeline — StateGraph wiring all three agents.

Defines the ScreenerState schema and the graph topology:
  screen → score → audit_bias → [conditional: release | human_review]

The conditional edge after audit_bias blocks shortlist release if
BiasReport.overall_risk == "high", routing to a human_review terminal
node instead.

Usage:
    graph = build_screening_graph(embedder, vector_store, llm_client)
    result = graph.invoke({"jd_text": "...", "top_k": 5})
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from agents.bias_auditor import BiasReport, FlaggedPattern, audit_bias, extract_reasoning_strings
from agents.scorer import CandidateScore, score_all_candidates
from agents.screener import extract_jd_criteria, screen_candidates
from embeddings import ResumeEmbedder
from logging_config import get_logger
from vectorstore import VectorStore

logger = get_logger(__name__)


# ── State Schema ───────────────────────────────────────────────────


class ScreenerState(TypedDict, total=False):
    """Shared state passed through the LangGraph pipeline.

    Attributes:
        jd_text: Raw job description text.
        jd_criteria: Criteria parsed from the JD during preprocessing.
        top_k: Number of candidates to shortlist.
        candidate_ids: All candidate IDs considered for this screening run.
        shortlisted_ids: Top-K candidate IDs from the screener agent.
        scores: Structured scores from the scorer agent.
        bias_report: Audit report from the bias auditor agent.
        release_blocked: True if bias audit blocked the release.
        error: Error message if any stage failed, else None.
    """

    jd_text: str
    jd_criteria: list[str]
    top_k: int
    candidate_ids: list[str]
    shortlisted_ids: list[str]
    scores: list[dict[str, Any]]  # Serialized CandidateScore dicts
    bias_report: dict[str, Any]  # Serialized BiasReport dict
    release_blocked: bool
    error: str | None


# ── Node Functions ─────────────────────────────────────────────────


def _make_screen_node(
    embedder: ResumeEmbedder,
    vector_store: VectorStore,
    llm_client: OpenAI,
) -> Any:
    """Create the screening node function.

    Args:
        embedder: Configured embedding client.
        vector_store: ChromaDB vector store.
        llm_client: OpenAI client for JD criteria extraction.

    Returns:
        Node function for the LangGraph graph.
    """

    def screen_node(state: ScreenerState) -> dict[str, Any]:
        """Screen candidates against the job description.

        Extracts criteria from JD, embeds it, queries ChromaDB,
        and returns top-K shortlisted candidate IDs.
        """
        try:
            jd_text = state["jd_text"]
            top_k = state.get("top_k", 5)

            logger.info("graph.screen.start", jd_length=len(jd_text), top_k=top_k)

            # Extract criteria from JD
            jd_criteria = extract_jd_criteria(jd_text, llm_client)

            # Screen candidates via semantic similarity
            screening_result = screen_candidates(
                jd_text=jd_text,
                embedder=embedder,
                vector_store=vector_store,
                top_k=top_k,
            )

            shortlisted_ids = [m.candidate_id for m in screening_result.shortlisted]
            all_candidate_ids = vector_store.get_candidate_ids()

            logger.info(
                "graph.screen.complete",
                total_candidates=len(all_candidate_ids),
                shortlisted=len(shortlisted_ids),
            )

            return {
                "jd_criteria": jd_criteria,
                "candidate_ids": all_candidate_ids,
                "shortlisted_ids": shortlisted_ids,
                "error": None,
            }

        except Exception as e:
            logger.error("graph.screen.error", error=str(e))
            return {
                "jd_criteria": [],
                "candidate_ids": [],
                "shortlisted_ids": [],
                "error": f"Screening failed: {e}",
            }

    return screen_node


def _make_score_node(
    vector_store: VectorStore,
    llm_client: OpenAI,
    model: str = "gpt-4o",
) -> Any:
    """Create the scoring node function.

    Args:
        vector_store: ChromaDB vector store for chunk retrieval.
        llm_client: OpenAI client for scoring LLM calls.
        model: LLM model to use.

    Returns:
        Node function for the LangGraph graph.
    """

    def score_node(state: ScreenerState) -> dict[str, Any]:
        """Score shortlisted candidates against JD criteria."""
        try:
            # Check for upstream errors
            if state.get("error"):
                return {"scores": [], "error": state["error"]}

            shortlisted_ids = state.get("shortlisted_ids", [])
            jd_criteria = state.get("jd_criteria", [])

            if not shortlisted_ids:
                logger.warning("graph.score.no_candidates")
                return {"scores": [], "error": "No candidates to score"}

            if not jd_criteria:
                logger.warning("graph.score.no_criteria")
                return {"scores": [], "error": "No JD criteria extracted"}

            logger.info(
                "graph.score.start",
                candidates=len(shortlisted_ids),
                criteria=len(jd_criteria),
            )

            scores = score_all_candidates(
                shortlisted_ids=shortlisted_ids,
                jd_criteria=jd_criteria,
                vector_store=vector_store,
                llm_client=llm_client,
                model=model,
            )

            # Serialize to dicts for state compatibility
            scores_dicts = [s.model_dump() for s in scores]

            logger.info("graph.score.complete", scored=len(scores))

            return {"scores": scores_dicts, "error": None}

        except Exception as e:
            logger.error("graph.score.error", error=str(e))
            return {"scores": [], "error": f"Scoring failed: {e}"}

    return score_node


def _make_audit_node(
    llm_client: OpenAI,
    model: str = "gpt-4o",
) -> Any:
    """Create the bias audit node function.

    Args:
        llm_client: OpenAI client for audit LLM call.
        model: LLM model to use.

    Returns:
        Node function for the LangGraph graph.
    """

    def audit_node(state: ScreenerState) -> dict[str, Any]:
        """Run the independent bias audit on scoring reasoning.

        CRITICAL: Only passes reasoning strings to the auditor.
        No scores, candidate IDs, or metadata are passed.
        """
        try:
            if state.get("error"):
                return {
                    "bias_report": BiasReport(
                        overall_risk="high",
                        flagged_patterns=[],
                        safe_to_proceed=False,
                        human_review_required=True,
                        summary=f"Audit skipped due to upstream error: {state['error']}",
                    ).model_dump(),
                    "release_blocked": True,
                    "error": state["error"],
                }

            scores_dicts = state.get("scores", [])
            if not scores_dicts:
                logger.warning("graph.audit.no_scores")
                return {
                    "bias_report": BiasReport(
                        overall_risk="low",
                        flagged_patterns=[],
                        safe_to_proceed=True,
                        human_review_required=False,
                        summary="No scores to audit. Pipeline produced no results.",
                    ).model_dump(),
                    "release_blocked": False,
                    "error": None,
                }

            # Reconstruct CandidateScore objects for reasoning extraction
            candidate_scores = [CandidateScore(**d) for d in scores_dicts]

            # HARD ISOLATION: Extract ONLY reasoning strings
            reasoning_strings = extract_reasoning_strings(candidate_scores)

            logger.info(
                "graph.audit.start",
                reasoning_count=len(reasoning_strings),
            )

            # Run the bias audit
            bias_report = audit_bias(
                reasoning_strings=reasoning_strings,
                llm_client=llm_client,
                model=model,
            )

            # Determine if release should be blocked
            release_blocked = bias_report.overall_risk == "high"

            logger.info(
                "graph.audit.complete",
                overall_risk=bias_report.overall_risk,
                release_blocked=release_blocked,
            )

            return {
                "bias_report": bias_report.model_dump(),
                "release_blocked": release_blocked,
                "error": None,
            }

        except Exception as e:
            logger.error("graph.audit.error", error=str(e))
            return {
                "bias_report": BiasReport(
                    overall_risk="high",
                    flagged_patterns=[],
                    safe_to_proceed=False,
                    human_review_required=True,
                    summary=f"Bias audit failed with error: {e}",
                ).model_dump(),
                "release_blocked": True,
                "error": f"Bias audit failed: {e}",
            }

    return audit_node


def release_node(state: ScreenerState) -> dict[str, Any]:
    """Terminal node: release results to the requester.

    Reached when bias audit risk is low or medium.
    """
    logger.info(
        "graph.release",
        shortlisted=len(state.get("shortlisted_ids", [])),
        scored=len(state.get("scores", [])),
        risk=state.get("bias_report", {}).get("overall_risk", "unknown"),
    )
    return {}


def human_review_node(state: ScreenerState) -> dict[str, Any]:
    """Terminal node: flag results for human review.

    Reached when bias audit risk is high. Results are NOT
    released automatically — a human must review and approve.
    """
    logger.warning(
        "graph.human_review_required",
        shortlisted=len(state.get("shortlisted_ids", [])),
        scored=len(state.get("scores", [])),
        risk=state.get("bias_report", {}).get("overall_risk", "unknown"),
        flagged_patterns=len(
            state.get("bias_report", {}).get("flagged_patterns", [])
        ),
    )
    return {}


# ── Conditional Edge ───────────────────────────────────────────────


def route_after_audit(state: ScreenerState) -> str:
    """Conditional edge: decide whether to release or require human review.

    Routes to human_review if:
      - BiasReport.overall_risk == "high"
      - release_blocked is True
      - There was an error in the pipeline

    Otherwise routes to release.

    Args:
        state: Current pipeline state.

    Returns:
        "human_review" or "release" — the name of the next node.
    """
    if state.get("error"):
        logger.info("graph.route.error_path", error=state["error"])
        return "human_review"

    release_blocked = state.get("release_blocked", False)
    bias_report = state.get("bias_report", {})
    overall_risk = bias_report.get("overall_risk", "high") if isinstance(bias_report, dict) else "high"

    if release_blocked or overall_risk == "high":
        logger.info("graph.route.human_review", risk=overall_risk)
        return "human_review"

    logger.info("graph.route.release", risk=overall_risk)
    return "release"


# ── Graph Builder ──────────────────────────────────────────────────


def build_screening_graph(
    embedder: ResumeEmbedder,
    vector_store: VectorStore,
    llm_client: Optional[OpenAI] = None,
    model: str = "gpt-4o",
) -> Any:
    """Build the full LangGraph StateGraph for resume screening.

    Graph topology:
        screen → score → audit_bias → [conditional] → release | human_review

    Args:
        embedder: Configured ResumeEmbedder for semantic search.
        vector_store: ChromaDB vector store with ingested resumes.
        llm_client: OpenAI client. Created from env if not provided.
        model: LLM model name for scoring and auditing.

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    if llm_client is None:
        from config import get_settings
        settings = get_settings()
        llm_client = OpenAI(api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY", ""))

    logger.info("graph.build.start", model=model)

    # Create the StateGraph
    graph = StateGraph(ScreenerState)

    # Add nodes
    graph.add_node("screen", _make_screen_node(embedder, vector_store, llm_client))
    graph.add_node("score", _make_score_node(vector_store, llm_client, model))
    graph.add_node("audit_bias", _make_audit_node(llm_client, model))
    graph.add_node("release", release_node)
    graph.add_node("human_review", human_review_node)

    # Set entry point
    graph.set_entry_point("screen")

    # Add edges: screen → score → audit_bias
    graph.add_edge("screen", "score")
    graph.add_edge("score", "audit_bias")

    # Conditional edge after audit: release or human_review
    graph.add_conditional_edges(
        "audit_bias",
        route_after_audit,
        {
            "release": "release",
            "human_review": "human_review",
        },
    )

    # Terminal nodes
    graph.add_edge("release", END)
    graph.add_edge("human_review", END)

    # Compile
    compiled = graph.compile()

    logger.info("graph.build.complete")

    return compiled
