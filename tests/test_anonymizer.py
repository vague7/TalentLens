"""Tests for preprocessing.anonymizer module.

Tests cover regex-based PII detection (emails, phones, URLs, years,
institutions) and the overall anonymize_text function.
NER tests are skipped if spaCy model is not available.
"""

from __future__ import annotations

import pytest

from preprocessing.anonymizer import AnonymizationResult, anonymize_text


class TestEmailAnonymization:
    """Tests for email detection and replacement."""

    def test_simple_email(self) -> None:
        """Standard email addresses should be replaced."""
        result = anonymize_text("Contact me at john.doe@gmail.com", use_ner=False)
        assert "[EMAIL]" in result.anonymized_text
        assert "john.doe@gmail.com" not in result.anonymized_text
        assert "john.doe@gmail.com" in result.replacements.get("email", [])

    def test_multiple_emails(self) -> None:
        """Multiple email addresses should all be replaced."""
        text = "Email: work@company.com or personal@outlook.com"
        result = anonymize_text(text, use_ner=False)
        assert result.anonymized_text.count("[EMAIL]") == 2
        assert len(result.replacements.get("email", [])) == 2

    def test_no_email(self) -> None:
        """Text without emails should be unchanged for email patterns."""
        result = anonymize_text("No contact info here", use_ner=False)
        assert "[EMAIL]" not in result.anonymized_text


class TestPhoneAnonymization:
    """Tests for phone number detection and replacement."""

    def test_us_phone(self) -> None:
        """US phone numbers should be detected."""
        result = anonymize_text("Call me at (555) 123-4567", use_ner=False)
        assert "[PHONE]" in result.anonymized_text
        assert "123-4567" not in result.anonymized_text

    def test_international_phone(self) -> None:
        """International phone numbers should be detected."""
        result = anonymize_text("Phone: +1-555-123-4567", use_ner=False)
        assert "[PHONE]" in result.anonymized_text

    def test_indian_phone(self) -> None:
        """Indian phone numbers should be detected."""
        result = anonymize_text("Mobile: +91 98765 43210", use_ner=False)
        assert "[PHONE]" in result.anonymized_text


class TestLinkedInAnonymization:
    """Tests for LinkedIn URL detection and replacement."""

    def test_linkedin_url(self) -> None:
        """LinkedIn profile URLs should be replaced."""
        result = anonymize_text(
            "LinkedIn: https://www.linkedin.com/in/john-doe-123/",
            use_ner=False,
        )
        assert "[LINKEDIN]" in result.anonymized_text
        assert "linkedin.com" not in result.anonymized_text

    def test_linkedin_without_https(self) -> None:
        """LinkedIn URLs without https should still be detected."""
        result = anonymize_text(
            "Profile: www.linkedin.com/in/johndoe",
            use_ner=False,
        )
        assert "[LINKEDIN]" in result.anonymized_text


class TestYearAnonymization:
    """Tests for year detection and replacement."""

    def test_graduation_year(self) -> None:
        """Graduation years should be replaced."""
        result = anonymize_text("Graduated in 2019", use_ner=False)
        assert "[YEAR]" in result.anonymized_text
        assert "2019" not in result.anonymized_text

    def test_year_range(self) -> None:
        """Year ranges should have both years replaced."""
        result = anonymize_text("Experience: 2018 - 2023", use_ner=False)
        assert result.anonymized_text.count("[YEAR]") == 2

    def test_non_year_numbers_preserved(self) -> None:
        """Numbers that aren't years (e.g., quantities) should not be replaced."""
        result = anonymize_text("Managed a team of 150 people", use_ner=False)
        assert "150" in result.anonymized_text  # Not a year


class TestInstitutionAnonymization:
    """Tests for institution/university detection."""

    def test_known_university(self) -> None:
        """Known university names should be replaced."""
        result = anonymize_text(
            "Graduated from Stanford with honors",
            use_ner=False,
        )
        assert "[INSTITUTION]" in result.anonymized_text
        assert "Stanford" not in result.anonymized_text

    def test_generic_university(self) -> None:
        """Generic 'university' keyword should trigger replacement."""
        result = anonymize_text(
            "Attended the University of Wisconsin",
            use_ner=False,
        )
        assert "[INSTITUTION]" in result.anonymized_text

    def test_iit_institution(self) -> None:
        """Indian institutions like IIT should be detected."""
        result = anonymize_text(
            "B.Tech from IIT Bombay",
            use_ner=False,
        )
        assert "[INSTITUTION]" in result.anonymized_text


class TestPhotoAnonymization:
    """Tests for photo reference detection."""

    def test_photo_reference(self) -> None:
        """Photo references should be replaced."""
        result = anonymize_text("See my profile picture attached", use_ner=False)
        assert "[PHOTO]" in result.anonymized_text


class TestAnonymizeText:
    """Integration tests for the full anonymize_text function."""

    def test_returns_anonymization_result(self) -> None:
        """Should return an AnonymizationResult dataclass."""
        result = anonymize_text("Simple text", use_ner=False)
        assert isinstance(result, AnonymizationResult)
        assert result.candidate_uuid  # Should have a UUID

    def test_custom_uuid(self) -> None:
        """Should use the provided candidate UUID."""
        result = anonymize_text("Text", candidate_uuid="test-uuid-123", use_ner=False)
        assert result.candidate_uuid == "test-uuid-123"

    def test_empty_input(self) -> None:
        """Empty input should return empty result gracefully."""
        result = anonymize_text("", use_ner=False)
        assert result.anonymized_text == ""
        assert result.replacement_count == 0

    def test_full_resume_snippet(self) -> None:
        """A realistic resume snippet should have multiple PII types replaced."""
        text = """
        John Doe
        john.doe@email.com | (555) 123-4567
        LinkedIn: https://www.linkedin.com/in/john-doe/

        Education
        Stanford University, BS Computer Science, 2019

        Experience
        Software Engineer at Google, 2019 - 2023
        """
        result = anonymize_text(text, use_ner=False)

        assert "[EMAIL]" in result.anonymized_text
        assert "[PHONE]" in result.anonymized_text
        assert "[LINKEDIN]" in result.anonymized_text
        assert "[YEAR]" in result.anonymized_text
        assert "[INSTITUTION]" in result.anonymized_text
        assert result.replacement_count > 0

    def test_replacement_tracking(self) -> None:
        """Replacements dict should track all found PII values."""
        text = "Email: test@example.com, Year: 2020"
        result = anonymize_text(text, use_ner=False)

        assert "email" in result.replacements
        assert "test@example.com" in result.replacements["email"]
        assert "year" in result.replacements
        assert "2020" in result.replacements["year"]

    def test_ner_graceful_degradation(self) -> None:
        """With NER enabled but model missing, should still do regex anonymization."""
        result = anonymize_text(
            "john@test.com in 2020",
            use_ner=True,
            spacy_model_name="nonexistent_model",
        )
        # Regex replacements should still work
        assert "[EMAIL]" in result.anonymized_text
        assert "[YEAR]" in result.anonymized_text
