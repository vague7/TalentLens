"""Section-aware chunker — Splits resume text into meaningful chunks.

Produces chunks aligned with resume sections (Experience, Education,
Skills, Projects, etc.) with metadata attached for downstream retrieval
and embedding.

Key design:
  - Detects standard resume section headers via regex
  - Falls back to fixed-size chunking with overlap if no sections detected
  - Each chunk carries metadata: section name, position, candidate_id
  - Configurable max_chunk_size and overlap
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from logging_config import get_logger

logger = get_logger(__name__)

# ── Section header patterns ────────────────────────────────────────

# Common resume section headers (case-insensitive)
SECTION_HEADERS: list[str] = [
    r"summary",
    r"objective",
    r"professional\s*summary",
    r"career\s*summary",
    r"profile",
    r"about\s*me",
    r"experience",
    r"work\s*experience",
    r"professional\s*experience",
    r"employment\s*history",
    r"work\s*history",
    r"education",
    r"academic\s*background",
    r"qualifications",
    r"skills",
    r"technical\s*skills",
    r"core\s*competencies",
    r"competencies",
    r"projects",
    r"personal\s*projects",
    r"key\s*projects",
    r"certifications",
    r"certificates",
    r"licenses?\s*(?:&|and)?\s*certifications?",
    r"awards",
    r"honors?\s*(?:&|and)?\s*awards?",
    r"achievements",
    r"publications",
    r"research",
    r"volunteer",
    r"volunteering",
    r"community\s*(?:service|involvement)",
    r"extracurricular",
    r"interests",
    r"hobbies",
    r"languages",
    r"references",
    r"additional\s*information",
]

# Build a combined regex: match a line that is primarily a section header
# Handles formats like "EXPERIENCE", "Experience:", "== EDUCATION ==", etc.
SECTION_PATTERN = re.compile(
    r"^\s*(?:[-=*_]{0,5}\s*)?"               # optional decorators
    r"(?:" + "|".join(SECTION_HEADERS) + r")"  # section name
    r"(?:\s*[:|\-—=*_]{0,5})?"                 # optional trailing decorators
    r"\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ResumeChunk:
    """A single chunk of resume text with metadata.

    Attributes:
        text: The chunk content.
        section: Detected section name (e.g., "Experience", "Education").
        chunk_index: Position of this chunk within the document (0-indexed).
        candidate_id: UUID of the candidate this chunk belongs to.
        char_count: Number of characters in the chunk.
        metadata: Additional key-value metadata.
    """

    text: str
    section: str
    chunk_index: int
    candidate_id: str
    char_count: int = 0
    metadata: dict[str, str | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute char_count after initialization."""
        self.char_count = len(self.text)


@dataclass
class ChunkResult:
    """Result of chunking a resume document.

    Attributes:
        candidate_id: UUID of the candidate.
        chunks: List of resume chunks.
        total_chunks: Number of chunks produced.
        sections_detected: List of section names found in the document.
        chunking_method: Either "section_aware" or "fixed_size".
    """

    candidate_id: str
    chunks: list[ResumeChunk] = field(default_factory=list)
    total_chunks: int = 0
    sections_detected: list[str] = field(default_factory=list)
    chunking_method: str = "section_aware"


def _detect_sections(text: str) -> list[tuple[str, int, int]]:
    """Detect section boundaries in resume text.

    Args:
        text: Full resume text.

    Returns:
        List of (section_name, start_pos, end_pos) tuples.
        end_pos is the start of the next section or end of text.
    """
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        return []

    sections: list[tuple[str, int, int]] = []
    for i, match in enumerate(matches):
        section_name = match.group().strip().strip("-=*_: ")
        # Normalize to title case
        section_name = section_name.title()

        start = match.end()  # Content starts after the header
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        sections.append((section_name, start, end))

    # If there's content before the first section header, capture it
    if matches and matches[0].start() > 0:
        preamble_text = text[: matches[0].start()].strip()
        if preamble_text:
            sections.insert(0, ("Header", 0, matches[0].start()))

    return sections


def _split_large_chunk(
    text: str,
    max_size: int,
    overlap: int,
) -> list[str]:
    """Split a large text block into overlapping sub-chunks.

    Args:
        text: Text to split.
        max_size: Maximum characters per sub-chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of sub-chunk strings.
    """
    if len(text) <= max_size:
        return [text]

    # Clamp overlap to prevent infinite loops (must advance each iteration)
    effective_overlap = min(overlap, max_size // 2)

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_size

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for the last paragraph break within the chunk
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + max_size // 2:
                end = para_break
            else:
                # Look for the last sentence break
                sentence_break = text.rfind(". ", start, end)
                if sentence_break > start + max_size // 2:
                    end = sentence_break + 1  # Include the period

        chunks.append(text[start:end].strip())
        start = end - effective_overlap

    return [c for c in chunks if c]  # Filter empty chunks


def chunk_resume(
    text: str,
    candidate_id: str,
    max_chunk_size: int = 1500,
    overlap: int = 200,
    min_chunk_size: int = 50,
) -> ChunkResult:
    """Split resume text into section-aware chunks with metadata.

    First attempts section-aware chunking by detecting resume section
    headers. Falls back to fixed-size chunking with overlap if no
    sections are detected.

    Args:
        text: Anonymized resume text.
        candidate_id: UUID of the candidate.
        max_chunk_size: Maximum characters per chunk.
        overlap: Overlap characters between consecutive chunks.
        min_chunk_size: Minimum characters for a chunk to be included.

    Returns:
        ChunkResult with list of ResumeChunk objects and metadata.
    """
    if not text or not text.strip():
        logger.warning("chunker.empty_input", candidate_id=candidate_id)
        return ChunkResult(candidate_id=candidate_id)

    logger.info(
        "chunker.start",
        candidate_id=candidate_id,
        text_length=len(text),
        max_chunk_size=max_chunk_size,
    )

    sections = _detect_sections(text)
    chunks: list[ResumeChunk] = []
    chunk_index = 0
    sections_detected: list[str] = []
    method = "section_aware"

    if sections:
        # Section-aware chunking
        # First, record ALL detected section names regardless of size
        sections_detected = [name for name, _, _ in sections]

        for section_idx, (section_name, start, end) in enumerate(sections):
            section_text = text[start:end].strip()
            if not section_text or len(section_text) < min_chunk_size:
                continue

            # Split large sections into sub-chunks
            sub_chunks = _split_large_chunk(section_text, max_chunk_size, overlap)

            for i, sub_text in enumerate(sub_chunks):
                chunk = ResumeChunk(
                    text=sub_text,
                    section=section_name,
                    chunk_index=chunk_index,
                    candidate_id=candidate_id,
                    metadata={
                        "sub_chunk": i,
                        "total_sub_chunks": len(sub_chunks),
                        "section_position": section_idx,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1
    else:
        # Fallback: fixed-size chunking
        method = "fixed_size"
        logger.info("chunker.fallback_fixed_size", candidate_id=candidate_id)

        sub_chunks = _split_large_chunk(text, max_chunk_size, overlap)
        for i, sub_text in enumerate(sub_chunks):
            if len(sub_text) < min_chunk_size:
                continue
            chunk = ResumeChunk(
                text=sub_text,
                section="General",
                chunk_index=i,
                candidate_id=candidate_id,
                metadata={"sub_chunk": i, "total_sub_chunks": len(sub_chunks)},
            )
            chunks.append(chunk)

    result = ChunkResult(
        candidate_id=candidate_id,
        chunks=chunks,
        total_chunks=len(chunks),
        sections_detected=sections_detected,
        chunking_method=method,
    )

    logger.info(
        "chunker.complete",
        candidate_id=candidate_id,
        total_chunks=len(chunks),
        sections=sections_detected,
        method=method,
    )

    return result
