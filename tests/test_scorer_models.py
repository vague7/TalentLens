"""Tests for agents.scorer pydantic models."""

from __future__ import annotations

import pytest
from agents.scorer import CandidateScore, CriterionScore


class TestCriterionScore:
    """Tests for the CriterionScore model validator."""

    def test_valid_criterion_score(self) -> None:
        """Evidence present in reasoning should pass validation."""
        score = CriterionScore(
            criterion="Python experience",
            score=8,
            reasoning='Candidate demonstrates strong Python skills: "5 years of Python development including Django and FastAPI"',
            evidence="5 years of Python development including Django and FastAPI",
        )
        assert score.score == 8
        assert score.evidence in score.reasoning

    def test_evidence_not_in_reasoning_raises(self) -> None:
        """Missing evidence in reasoning must raise ValueError."""
        with pytest.raises(ValueError, match="reasoning must contain the evidence quote"):
            CriterionScore(
                criterion="Python experience",
                score=7,
                reasoning="Candidate has some Python experience.",
                evidence="5 years of Python development",
            )

    def test_score_bounds(self) -> None:
        """Score must be between 1 and 10."""
        with pytest.raises(ValueError):
            CriterionScore(
                criterion="Test",
                score=0,
                reasoning='Evidence: "test quote"',
                evidence="test quote",
            )
        with pytest.raises(ValueError):
            CriterionScore(
                criterion="Test",
                score=11,
                reasoning='Evidence: "test quote"',
                evidence="test quote",
            )


class TestCandidateScore:
    """Tests for the CandidateScore model."""

    def test_valid_candidate_score(self) -> None:
        """A well-formed CandidateScore should validate successfully."""
        criterion = CriterionScore(
            criterion="Python",
            score=8,
            reasoning='Strong skills: "5 years Python"',
            evidence="5 years Python",
        )
        candidate = CandidateScore(
            candidate_id="uuid-1234",
            overall_score=8.0,
            criteria=[criterion],
            recommendation="shortlist",
            confidence=0.9,
            processing_time_ms=450,
        )
        assert candidate.recommendation == "shortlist"

    def test_invalid_recommendation(self) -> None:
        """Recommendation must be one of shortlist/hold/reject."""
        criterion = CriterionScore(
            criterion="Python",
            score=8,
            reasoning='Strong skills: "5 years Python"',
            evidence="5 years Python",
        )
        with pytest.raises(ValueError):
            CandidateScore(
                candidate_id="uuid-1234",
                overall_score=8.0,
                criteria=[criterion],
                recommendation="maybe",  # type: ignore[arg-type]
                confidence=0.9,
                processing_time_ms=450,
            )
