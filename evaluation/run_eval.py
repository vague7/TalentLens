"""CLI Eval Runner — Executes RAGAS evaluation on the synthetic dataset.

This script acts as the CI/CD test gate. It runs RAGAS evaluation
on the synthetic dataset and asserts that all metrics exceed the
configured thresholds (default 0.8).

Usage:
    python -m evaluation.run_eval --threshold 0.85
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from evaluation.metrics import evaluate_pipeline
from logging_config import get_logger

logger = get_logger(__name__)

# Load env vars for OpenAI access
load_dotenv()


def run_evaluation(data_path: Path, threshold: float = 0.8) -> bool:
    """Run RAGAS evaluation and check against thresholds.

    Args:
        data_path: Path to the synthetic dataset JSON file.
        threshold: Minimum required score for all aggregate metrics.

    Returns:
        True if all metrics pass the threshold, False otherwise.
    """
    logger.info("run_eval.start", dataset=str(data_path), threshold=threshold)

    if not data_path.exists():
        logger.error("run_eval.missing_data", path=str(data_path))
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Initialize LangChain wrappers for RAGAS
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        logger.error("run_eval.missing_deps", error="Install langchain-openai")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("run_eval.missing_key", error="OPENAI_API_KEY not set")
        sys.exit(1)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    # Run evaluation
    try:
        results = evaluate_pipeline(eval_data, llm_client=llm, embedder=embedder)
    except Exception as e:
        logger.error("run_eval.fatal_error", error=str(e))
        sys.exit(1)

    # Print results
    print("\n" + "=" * 40)
    print("📊 RAGAS EVALUATION RESULTS")
    print("=" * 40)
    print(f"Faithfulness:       {results.faithfulness:.3f} " + ("✅" if results.faithfulness >= threshold else "❌"))
    print(f"Answer Relevancy:   {results.answer_relevancy:.3f} " + ("✅" if results.answer_relevancy >= threshold else "❌"))
    print(f"Context Precision:  {results.context_precision:.3f} " + ("✅" if results.context_precision >= threshold else "❌"))
    print(f"Processing Time:    {results.processing_time_ms} ms")
    print("=" * 40 + "\n")

    # Save detailed results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "eval_results.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "aggregate": {
                    "faithfulness": results.faithfulness,
                    "answer_relevancy": results.answer_relevancy,
                    "context_precision": results.context_precision,
                },
                "threshold": threshold,
                "details": results.details,
            },
            f,
            indent=2,
        )
    logger.info("run_eval.saved_results", path=str(out_file))

    # Assert thresholds
    passed = (
        results.faithfulness >= threshold
        and results.answer_relevancy >= threshold
        and results.context_precision >= threshold
    )

    if passed:
        logger.info("run_eval.passed", threshold=threshold)
        return True
    else:
        logger.error("run_eval.failed", threshold=threshold)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS Evaluation Pipeline")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum required score for metrics (0.0 to 1.0)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="evaluation/test_data/synthetic.json",
        help="Path to synthetic dataset",
    )
    args = parser.parse_args()

    data_file = Path(args.data)
    success = run_evaluation(data_file, args.threshold)

    if not success:
        sys.exit(1)
    sys.exit(0)
