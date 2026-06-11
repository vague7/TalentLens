"""Agent 3 — Independent Bias Auditor.

Receives ONLY reasoning strings from the scorer — NO scores, candidate IDs,
similarity scores, or metadata. This hard isolation ensures the auditor
evaluates reasoning quality without anchoring on numeric outcomes.

Design:
  - Input: list[str] of reasoning strings ONLY
  - Uses the exact bias auditor system prompt from the spec
  - Returns structured BiasReport via JSON output
  - All LLM calls: temperature=0, retry, response time logging
"""

from __future__ import annotations

import json
import time
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from logging_config import get_logger

logger = get_logger(__name__)


# ── Pydantic Output Models ─────────────────────────────────────────


class FlaggedPattern(BaseModel):
    """A single bias pattern detected in the scoring reasoning.

    Attributes:
        pattern: Category of bias detected.
        frequency: Number of occurrences across all reasoning strings.
        example_quote: Verbatim quote illustrating the pattern.
        recommendation: Suggested remediation action.
    """

    pattern: str
    frequency: int
    example_quote: str
    recommendation: str


class BiasReport(BaseModel):
    """Structured output from the bias auditor agent.

    Attributes:
        overall_risk: Aggregate risk level for the scoring batch.
        flagged_patterns: List of detected bias patterns.
        safe_to_proceed: Whether results can be released automatically.
        human_review_required: Whether a human must review before release.
        summary: Two-sentence summary of findings.
    """

    overall_risk: str  # "low" | "medium" | "high"
    flagged_patterns: list[FlaggedPattern]
    safe_to_proceed: bool
    human_review_required: bool
    summary: str  # 2 sentences max


# ── Bias Auditor System Prompt (exact spec) ────────────────────────


BIAS_AUDITOR_SYSTEM_PROMPT = """You are an independent AI auditor reviewing hiring decision reasoning \
for fairness. You receive reasoning strings from a scoring system. \
You have NO access to candidate names, scores, or identifiers.

Analyze the collective reasoning and flag any of the following:

1. Proxy discrimination — language that correlates with protected \
attributes (employment gaps penalized, school prestige rewarded \
beyond JD requirements, name-adjacent cultural signals).

2. Inconsistent criterion application — the same skill is evaluated \
more favorably in some reasoning strings than others without \
justification.

3. Irrelevant penalization — negative signals for things not in the \
JD requirements.

4. Gendered language — adjectives with demographic connotation \
(aggressive, nurturing, bossy) used as positive/negative signals.

5. Credential inflation — degree required in reasoning when JD \
specifies 'or equivalent experience'.

Return ONLY a valid JSON object matching this schema:
{
  "overall_risk": "low" | "medium" | "high",
  "flagged_patterns": [
    {
      "pattern": str,
      "frequency": int,
      "example_quote": str,
      "recommendation": str
    }
  ],
  "safe_to_proceed": bool,
  "human_review_required": bool,
  "summary": str
}"""


# ── Bias Auditor Agent ─────────────────────────────────────────────


def audit_bias(
    reasoning_strings: list[str],
    llm_client: OpenAI,
    model: str = "gpt-4o",
) -> BiasReport:
    """Run the independent bias audit on scoring reasoning strings.

    CRITICAL ISOLATION: This function receives ONLY reasoning strings.
    It must NOT receive: numeric scores, candidate_ids, similarity
    scores, or any metadata. This is a hard isolation requirement.

    Args:
        reasoning_strings: List of reasoning strings from the scorer.
            These are the CriterionScore.reasoning values ONLY.
        llm_client: OpenAI client for the audit LLM call.
        model: LLM model to use.

    Returns:
        BiasReport with risk assessment and flagged patterns.
    """
    start_time = time.perf_counter()

    logger.info(
        "bias_auditor.start",
        reasoning_count=len(reasoning_strings),
    )

    # Format reasoning strings for the auditor — numbered, no metadata
    formatted_reasoning = "\n\n".join(
        f"Reasoning {i+1}:\n\"{reasoning}\""
        for i, reasoning in enumerate(reasoning_strings)
    )

    user_prompt = f"""Below are {len(reasoning_strings)} reasoning strings from a resume scoring system. \
Analyze them collectively for bias patterns.

{formatted_reasoning}"""

    # Call the bias auditor LLM
    report = _call_bias_auditor_llm(
        llm_client=llm_client,
        model=model,
        system_prompt=BIAS_AUDITOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    logger.info(
        "bias_auditor.complete",
        overall_risk=report.overall_risk,
        flagged_count=len(report.flagged_patterns),
        safe_to_proceed=report.safe_to_proceed,
        human_review_required=report.human_review_required,
        processing_time_ms=elapsed_ms,
    )

    return report


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _call_bias_auditor_llm(
    llm_client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> BiasReport:
    """Call GPT-4o for bias auditing with structured JSON output.

    Args:
        llm_client: OpenAI client instance.
        model: Model name.
        system_prompt: The bias auditor system prompt (exact spec).
        user_prompt: Formatted reasoning strings.

    Returns:
        Parsed BiasReport.
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
    logger.info("bias_auditor.llm_call", model=model, response_time_ms=elapsed_ms)

    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)

    return BiasReport(**parsed)


def extract_reasoning_strings(
    scores: list,
) -> list[str]:
    """Extract ONLY reasoning strings from CandidateScore objects.

    This function enforces the hard isolation requirement: the bias
    auditor receives only reasoning text, never scores, IDs, or metadata.

    Args:
        scores: List of CandidateScore objects from the scorer.

    Returns:
        List of reasoning strings (one per criterion per candidate).
    """
    reasoning_strings: list[str] = []

    for candidate_score in scores:
        for criterion_score in candidate_score.criteria:
            reasoning_strings.append(criterion_score.reasoning)

    logger.info(
        "bias_auditor.extracted_reasoning",
        total_strings=len(reasoning_strings),
        # NOTE: We deliberately do NOT log candidate_ids or scores here
    )

    return reasoning_strings
