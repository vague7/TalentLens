"""Agents package — Multi-agent pipeline components.

Contains three agents:
  - screener: RAG retrieval + top-K shortlisting
  - scorer: Structured LLM scoring with pydantic models
  - bias_auditor: Independent bias audit (isolated from scores/IDs)

Public API:
  from agents.screener import screen_candidates, extract_jd_criteria
  from agents.scorer import CandidateScore, CriterionScore, score_all_candidates
  from agents.bias_auditor import BiasReport, FlaggedPattern, audit_bias
"""

from agents.bias_auditor import BiasReport, FlaggedPattern, audit_bias
from agents.scorer import CandidateScore, CriterionScore, score_all_candidates
from agents.screener import ScreeningResult, screen_candidates

__all__ = [
    "BiasReport",
    "CandidateScore",
    "CriterionScore",
    "FlaggedPattern",
    "ScreeningResult",
    "audit_bias",
    "score_all_candidates",
    "screen_candidates",
]
