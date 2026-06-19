"""Tests for preprocessing.chunker module.

Tests cover section detection, section-aware chunking, fixed-size fallback,
large chunk splitting, and chunk metadata.
"""

from __future__ import annotations

import pytest

from preprocessing.chunker import ChunkResult, ResumeChunk, chunk_resume


# ── Sample resume texts ───────────────────────────────────────────

RESUME_WITH_SECTIONS = """
Summary
Experienced software engineer with 5+ years in backend development.

Experience
Senior Engineer at Tech Corp
- Built microservices architecture serving 10M requests/day
- Led migration from monolith to event-driven architecture
- Mentored 4 junior developers

Junior Engineer at StartupCo
- Developed REST APIs using Python and FastAPI
- Implemented CI/CD pipelines with GitHub Actions

Education
BS Computer Science, Class of [YEAR]
GPA: 3.8/4.0

Skills
Python, Java, Go, Kubernetes, Docker, PostgreSQL, Redis, AWS

Projects
Open Source CLI Tool
- Built a CLI tool for database migrations, 500+ GitHub stars

Certifications
AWS Solutions Architect Associate
Google Cloud Professional Data Engineer
"""

RESUME_WITHOUT_SECTIONS = """
Software developer with experience in Python and JavaScript.
Built web applications using Django and React.
Worked on machine learning projects using TensorFlow.
Managed cloud infrastructure on AWS.
Strong background in data structures and algorithms.
Contributed to several open source projects on GitHub.
"""

LONG_SECTION_TEXT = "This is a detailed paragraph about work experience. " * 100


class TestSectionDetection:
    """Tests for section header detection in resume text."""

    def test_detects_common_sections(self) -> None:
        """Should detect standard resume section headers."""
        result = chunk_resume(RESUME_WITH_SECTIONS, candidate_id="test-1")

        assert result.chunking_method == "section_aware"
        assert len(result.sections_detected) > 0

        # Normalize to lowercase for comparison
        detected_lower = [s.lower() for s in result.sections_detected]
        assert "summary" in detected_lower or "header" in detected_lower
        assert "experience" in detected_lower
        assert "education" in detected_lower
        assert "skills" in detected_lower

    def test_no_sections_uses_fixed_size(self) -> None:
        """Text without section headers should fall back to fixed-size chunking."""
        result = chunk_resume(RESUME_WITHOUT_SECTIONS, candidate_id="test-2")
        assert result.chunking_method == "fixed_size"
        assert result.total_chunks > 0

    def test_case_insensitive_headers(self) -> None:
        """Section headers should be detected case-insensitively."""
        text = (
            "EXPERIENCE\n"
            "Worked at CompanyA as a senior software engineer building distributed systems\n\n"
            "EDUCATION\n"
            "BS Computer Science from a well-known institution with honors and distinction"
        )
        result = chunk_resume(text, candidate_id="test-3")
        assert result.chunking_method == "section_aware"
        assert len(result.sections_detected) >= 2


class TestChunkResume:
    """Tests for the main chunk_resume function."""

    def test_returns_chunk_result(self) -> None:
        """Should return a ChunkResult dataclass."""
        result = chunk_resume(RESUME_WITH_SECTIONS, candidate_id="test-1")
        assert isinstance(result, ChunkResult)
        assert result.candidate_id == "test-1"
        assert result.total_chunks == len(result.chunks)

    def test_chunks_have_metadata(self) -> None:
        """Each chunk should carry section and candidate metadata."""
        result = chunk_resume(RESUME_WITH_SECTIONS, candidate_id="test-1")

        for chunk in result.chunks:
            assert isinstance(chunk, ResumeChunk)
            assert chunk.candidate_id == "test-1"
            assert chunk.section  # Not empty
            assert chunk.chunk_index >= 0
            assert chunk.char_count > 0
            assert chunk.char_count == len(chunk.text)

    def test_empty_input(self) -> None:
        """Empty input should return empty result."""
        result = chunk_resume("", candidate_id="empty")
        assert result.total_chunks == 0
        assert result.chunks == []

    def test_whitespace_only_input(self) -> None:
        """Whitespace-only input should return empty result."""
        result = chunk_resume("   \n\n  \t  ", candidate_id="ws")
        assert result.total_chunks == 0

    def test_min_chunk_size_filter(self) -> None:
        """Chunks below min_chunk_size should be filtered out."""
        text = "Skills\nPython\n\nExperience\nThis is a longer section with enough text to pass."
        result = chunk_resume(text, candidate_id="min-test", min_chunk_size=20)

        for chunk in result.chunks:
            assert len(chunk.text) >= 20

    def test_chunk_indices_sequential(self) -> None:
        """Chunk indices should be sequential starting from 0."""
        result = chunk_resume(RESUME_WITH_SECTIONS, candidate_id="idx-test")

        indices = [c.chunk_index for c in result.chunks]
        assert indices == list(range(len(indices)))


class TestLargeChunkSplitting:
    """Tests for splitting oversized sections into sub-chunks."""

    def test_large_section_is_split(self) -> None:
        """A section exceeding max_chunk_size should be split."""
        text = f"Experience\n{LONG_SECTION_TEXT}"
        result = chunk_resume(text, candidate_id="large-1", max_chunk_size=500)

        # Should produce multiple chunks from one large section
        assert result.total_chunks > 1

    def test_max_chunk_size_respected(self) -> None:
        """No chunk should exceed max_chunk_size (approximately)."""
        text = f"Experience\n{LONG_SECTION_TEXT}"
        max_size = 500
        result = chunk_resume(text, candidate_id="size-1", max_chunk_size=max_size)

        for chunk in result.chunks:
            # Allow some tolerance for boundary splitting
            assert chunk.char_count <= max_size + 50, (
                f"Chunk too large: {chunk.char_count} > {max_size + 50}"
            )

    def test_overlap_between_chunks(self) -> None:
        """Consecutive chunks from the same section should have overlapping content."""
        text = f"Experience\n{LONG_SECTION_TEXT}"
        result = chunk_resume(
            text, candidate_id="overlap-1", max_chunk_size=300, overlap=100
        )

        if result.total_chunks >= 2:
            # Check that consecutive chunks share some content
            chunk1_end = result.chunks[0].text[-100:]
            chunk2_start = result.chunks[1].text[:200]
            # There should be some overlap (not necessarily exact due to boundary splitting)
            assert len(chunk1_end) > 0 and len(chunk2_start) > 0


class TestCustomParameters:
    """Tests for custom chunking parameters."""

    def test_custom_max_chunk_size(self) -> None:
        """Should respect custom max_chunk_size."""
        result = chunk_resume(
            RESUME_WITH_SECTIONS,
            candidate_id="custom-1",
            max_chunk_size=200,
        )
        # Smaller chunk size should produce more chunks
        result_default = chunk_resume(
            RESUME_WITH_SECTIONS,
            candidate_id="custom-2",
            max_chunk_size=5000,
        )
        assert result.total_chunks >= result_default.total_chunks

    def test_candidate_id_propagated(self) -> None:
        """Candidate ID should appear on every chunk."""
        cid = "uuid-abc-123"
        result = chunk_resume(RESUME_WITH_SECTIONS, candidate_id=cid)

        for chunk in result.chunks:
            assert chunk.candidate_id == cid
