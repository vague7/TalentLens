"""RAGAS Evaluation Layer — Metric computation for the Screener and Scorer.

Uses the RAGAS framework to evaluate the multi-agent pipeline on three core metrics:
  1. Faithfulness: Is the scorer's reasoning grounded in the retrieved chunks?
  2. Answer Relevancy: Does the reasoning properly address the JD criteria?
  3. Context Precision: Were the most relevant resume chunks retrieved at the top?

Usage:
    metrics = evaluate_pipeline(test_dataset, llm_client, embedder)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset
from logging_config import get_logger

# Import RAGAS metrics
try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    
    # Check if we need to use the newer RAGAS API
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        RAGAS_V0_4 = True
    except ImportError:
        RAGAS_V0_4 = False

except ImportError:
    raise ImportError(
        "RAGAS is not installed. Please install with `pip install ragas datasets pandas`."
    )

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Aggregated results from a RAGAS evaluation run.

    Attributes:
        faithfulness: Score from 0.0 to 1.0.
        answer_relevancy: Score from 0.0 to 1.0.
        context_precision: Score from 0.0 to 1.0.
        processing_time_ms: Time taken to run evaluation.
        details: Full row-by-row metric data.
    """

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    processing_time_ms: int
    details: list[dict[str, Any]] = field(default_factory=list)


def evaluate_pipeline(
    eval_data: list[dict[str, Any]],
    llm_client: Any = None,
    embedder: Any = None,
) -> EvaluationResult:
    """Run RAGAS evaluation on a batch of test cases.

    Args:
        eval_data: List of dicts, each containing:
            - question: The JD criteria being evaluated
            - contexts: List of retrieved resume chunks (strings)
            - answer: The scorer's reasoning text
            - ground_truth: Expected ideal reasoning/recommendation (optional but required for some metrics)
        llm_client: LangChain LLM instance to use as the judge (e.g. ChatOpenAI).
        embedder: LangChain Embeddings instance.

    Returns:
        EvaluationResult containing aggregate and detailed scores.
    """
    start_time = time.perf_counter()
    logger.info("evaluation.start", count=len(eval_data))

    # Standardize dataset for RAGAS (expects specific column names)
    formatted_data = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": [],
    }

    for item in eval_data:
        formatted_data["question"].append(item.get("question", ""))
        formatted_data["contexts"].append(item.get("contexts", []))
        formatted_data["answer"].append(item.get("answer", ""))
        formatted_data["ground_truth"].append(item.get("ground_truth", ""))

    dataset = Dataset.from_dict(formatted_data)

    metrics_list = [faithfulness, answer_relevancy, context_precision]

    try:
        # Wrap the LLM and Embedder if required by newer RAGAS versions
        # Or fall back to older versions. If neither provided, RAGAS defaults to OpenAI env vars.
        kwargs = {}
        if llm_client and embedder:
            if RAGAS_V0_4:
                kwargs["llm"] = LangchainLLMWrapper(llm_client)
                kwargs["embeddings"] = LangchainEmbeddingsWrapper(embedder)
            else:
                kwargs["llm"] = llm_client
                kwargs["embeddings"] = embedder

        # We set raise_exceptions=False to avoid crashing the whole suite on one bad row
        result = evaluate(
            dataset=dataset,
            metrics=metrics_list,
            raise_exceptions=False,
            **kwargs,
        )
    except Exception as e:
        logger.error("evaluation.failed", error=str(e))
        raise RuntimeError(f"RAGAS evaluation failed: {e}") from e

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    # Result is a ragas Result object, which behaves like a dict for aggregate scores
    # and has a .to_pandas() method for details.
    df = result.to_pandas()
    details = df.to_dict(orient="records")

    # Handle NaNs
    import math

    def safe_get(val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 0.0
        return float(val)

    agg_faithfulness = safe_get(result.get("faithfulness", 0.0))
    agg_relevancy = safe_get(result.get("answer_relevancy", 0.0))
    agg_precision = safe_get(result.get("context_precision", 0.0))

    logger.info(
        "evaluation.complete",
        faithfulness=agg_faithfulness,
        answer_relevancy=agg_relevancy,
        context_precision=agg_precision,
        processing_time_ms=elapsed_ms,
    )

    return EvaluationResult(
        faithfulness=agg_faithfulness,
        answer_relevancy=agg_relevancy,
        context_precision=agg_precision,
        processing_time_ms=elapsed_ms,
        details=details,
    )
