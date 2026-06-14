"""Anonymizer — PII stripping for resume text.

Strips personally identifiable information before any agent sees a document.
Uses regex patterns for structured PII (emails, phones, URLs, years) and
spaCy NER for unstructured PII (names, organizations/institutions).

Replacements:
    - Full names       → [CANDIDATE_UUID]
    - Emails           → [EMAIL]
    - Phone numbers    → [PHONE]
    - LinkedIn URLs    → [LINKEDIN]
    - Graduation years → [YEAR]
    - Institutions     → [INSTITUTION]
    - Photo references → [PHOTO]

This runs BEFORE any agent sees a document — non-negotiable.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from logging_config import get_logger

logger = get_logger(__name__)

# ── Regex patterns for structured PII ──────────────────────────────

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"       # optional country code
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"     # optional area code
    r"\d{3,4}[\s\-.]?\d{3,4}"         # main number
)

LINKEDIN_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+/?",
    re.IGNORECASE,
)

# Match URLs in general (http/https)
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+",
    re.IGNORECASE,
)

# Match 4-digit years (1950–2039) that are likely graduation/employment years
YEAR_PATTERN = re.compile(
    r"\b(19[5-9]\d|20[0-3]\d)\b"
)

# Photo/image references in documents
PHOTO_PATTERN = re.compile(
    r"\b(?:photo|photograph|headshot|profile\s*(?:pic|picture|image)|avatar)\b",
    re.IGNORECASE,
)

# ── Well-known institutions (expandable list) ─────────────────────

KNOWN_INSTITUTIONS: set[str] = {
    # US Universities
    "harvard", "stanford", "mit", "yale", "princeton", "columbia",
    "caltech", "uchicago", "university of chicago", "upenn",
    "university of pennsylvania", "cornell", "duke", "northwestern",
    "johns hopkins", "brown", "vanderbilt", "rice", "notre dame",
    "georgetown", "emory", "carnegie mellon", "uc berkeley",
    "university of california", "ucla", "usc", "nyu", "umich",
    "university of michigan", "georgia tech", "ut austin",
    "university of texas", "uiuc", "purdue", "ohio state",
    "penn state", "university of washington", "uw madison",
    # UK Universities
    "oxford", "cambridge", "imperial college", "ucl", "lse",
    "london school of economics", "edinburgh", "manchester",
    "kings college", "king's college",
    # Indian Universities/Institutions
    "iit", "iim", "iisc", "nit", "bits", "bits pilani",
    "iit bombay", "iit delhi", "iit madras", "iit kanpur",
    "iim ahmedabad", "iim bangalore", "iiit",
    # General patterns
    "university", "college", "institute", "school of",
    "academy", "polytechnic",
}

# Compiled pattern for institution detection
INSTITUTION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(inst) for inst in KNOWN_INSTITUTIONS) + r")\b",
    re.IGNORECASE,
)


@dataclass
class AnonymizationResult:
    """Result of anonymizing a document.

    Attributes:
        anonymized_text: The text with all PII replaced by placeholders.
        candidate_uuid: UUID assigned to this candidate.
        replacements: Map of placeholder type → list of original values found.
        replacement_count: Total number of replacements made.
    """

    anonymized_text: str
    candidate_uuid: str
    replacements: dict[str, list[str]] = field(default_factory=dict)
    replacement_count: int = 0


def _apply_regex_replacements(
    text: str,
    replacements: dict[str, list[str]],
) -> tuple[str, dict[str, list[str]], int]:
    """Apply regex-based PII replacements.

    Args:
        text: The input text to anonymize.
        replacements: Dict to accumulate found PII values.

    Returns:
        Tuple of (modified text, updated replacements dict, replacement count).
    """
    count = 0

    # Order matters: LinkedIn before general URLs, emails before phones

    # LinkedIn URLs
    for match in LINKEDIN_PATTERN.findall(text):
        replacements.setdefault("linkedin", []).append(match)
        count += 1
    text = LINKEDIN_PATTERN.sub("[LINKEDIN]", text)

    # General URLs (after LinkedIn to avoid double-matching)
    for match in URL_PATTERN.findall(text):
        if "linkedin.com" not in match.lower():
            replacements.setdefault("url", []).append(match)
            count += 1
    text = URL_PATTERN.sub(
        lambda m: "[LINKEDIN]" if "linkedin.com" in m.group().lower() else "[URL]",
        text,
    )

    # Emails
    for match in EMAIL_PATTERN.findall(text):
        replacements.setdefault("email", []).append(match)
        count += 1
    text = EMAIL_PATTERN.sub("[EMAIL]", text)

    # Phone numbers (after emails to avoid matching numbers inside emails)
    for match in PHONE_PATTERN.findall(text):
        # Filter out very short matches that are likely not phone numbers
        digits_only = re.sub(r"\D", "", match)
        if len(digits_only) >= 7:
            replacements.setdefault("phone", []).append(match)
            count += 1
    # Only replace phone-length matches
    def _replace_phone(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group())
        if len(digits) >= 7:
            return "[PHONE]"
        return m.group()
    text = PHONE_PATTERN.sub(_replace_phone, text)

    # Photo references
    for match in PHOTO_PATTERN.findall(text):
        replacements.setdefault("photo", []).append(match)
        count += 1
    text = PHOTO_PATTERN.sub("[PHOTO]", text)

    # Years
    for match in YEAR_PATTERN.findall(text):
        replacements.setdefault("year", []).append(match)
        count += 1
    text = YEAR_PATTERN.sub("[YEAR]", text)

    # Known institutions
    for match in INSTITUTION_PATTERN.findall(text):
        replacements.setdefault("institution", []).append(match)
        count += 1
    text = INSTITUTION_PATTERN.sub("[INSTITUTION]", text)

    return text, replacements, count


def _apply_ner_replacements(
    text: str,
    replacements: dict[str, list[str]],
    candidate_uuid: str,
    spacy_model_name: str = "en_core_web_trf",
) -> tuple[str, dict[str, list[str]], int]:
    """Apply spaCy NER-based PII replacements for names and organizations.

    Args:
        text: The input text (already regex-anonymized).
        replacements: Dict to accumulate found PII values.
        candidate_uuid: UUID to replace person names with.
        spacy_model_name: Name of the spaCy model to load.

    Returns:
        Tuple of (modified text, updated replacements dict, replacement count).
    """
    try:
        import spacy
        nlp = spacy.load(spacy_model_name)
    except (ImportError, OSError):
        logger.warning(
            "anonymizer.spacy.unavailable",
            model=spacy_model_name,
            msg="spaCy model not available; skipping NER-based anonymization",
        )
        return text, replacements, 0

    doc = nlp(text)
    count = 0

    # Sort entities by start position (descending) to replace from end to start
    # This preserves character positions during replacement
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)

    for ent in entities:
        if ent.label_ == "PERSON":
            original = ent.text
            # Skip if it's already a placeholder
            if original.startswith("[") and original.endswith("]"):
                continue
            replacements.setdefault("person", []).append(original)
            text = text[:ent.start_char] + f"[CANDIDATE_{candidate_uuid}]" + text[ent.end_char:]
            count += 1
        elif ent.label_ == "ORG":
            original = ent.text
            if original.startswith("[") and original.endswith("]"):
                continue
            # Check if it's likely an educational institution
            if any(kw in original.lower() for kw in ("university", "college", "institute", "school", "academy")):
                replacements.setdefault("institution", []).append(original)
                text = text[:ent.start_char] + "[INSTITUTION]" + text[ent.end_char:]
                count += 1

    return text, replacements, count


def anonymize_text(
    text: str,
    candidate_uuid: Optional[str] = None,
    use_ner: bool = True,
    spacy_model_name: str = "en_core_web_trf",
) -> AnonymizationResult:
    """Anonymize resume text by replacing PII with placeholders.

    This function applies two layers of anonymization:
    1. Regex-based: emails, phones, LinkedIn URLs, years, institutions, photos
    2. NER-based (optional): person names, organization names

    Args:
        text: Raw resume text to anonymize.
        candidate_uuid: Optional UUID for this candidate. Auto-generated if None.
        use_ner: Whether to apply spaCy NER-based anonymization.
        spacy_model_name: spaCy model to use for NER.

    Returns:
        AnonymizationResult with anonymized text, UUID, and replacement log.
    """
    if not text or not text.strip():
        logger.warning("anonymizer.empty_input")
        return AnonymizationResult(
            anonymized_text="",
            candidate_uuid=candidate_uuid or str(uuid.uuid4()),
            replacements={},
            replacement_count=0,
        )

    cid = candidate_uuid or str(uuid.uuid4())
    logger.info("anonymizer.start", candidate_uuid=cid, text_length=len(text))

    replacements: dict[str, list[str]] = {}
    total_count = 0

    # Layer 1: Regex-based replacements
    anonymized, replacements, regex_count = _apply_regex_replacements(text, replacements)
    total_count += regex_count

    # Layer 2: NER-based replacements (optional — requires spaCy model)
    if use_ner:
        anonymized, replacements, ner_count = _apply_ner_replacements(
            anonymized, replacements, cid, spacy_model_name
        )
        total_count += ner_count

    logger.info(
        "anonymizer.complete",
        candidate_uuid=cid,
        total_replacements=total_count,
        categories=list(replacements.keys()),
    )

    return AnonymizationResult(
        anonymized_text=anonymized,
        candidate_uuid=cid,
        replacements=replacements,
        replacement_count=total_count,
    )
