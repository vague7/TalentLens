"""Evaluation Dashboard — Visualizes RAGAS metrics and Bias audit flags.

Provides a Streamlit interface to view the results of the evaluation
runs stored in `results/eval_results.json`.

Usage:
    streamlit run evaluation/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Resume Screener | Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Multi-Agent Pipeline Evaluation")
st.markdown(
    "Visualizing RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision) "
    "and Bias Auditor flags across synthetic test runs."
)


@st.cache_data
def load_eval_data() -> dict:
    """Load the latest evaluation results."""
    data_path = Path("results/eval_results.json")
    if not data_path.exists():
        return {}
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    data = load_eval_data()

    if not data:
        st.warning("No evaluation results found. Run `python -m evaluation.run_eval` first.")
        return

    aggregate = data.get("aggregate", {})
    threshold = data.get("threshold", 0.8)
    details = data.get("details", [])

    # ── Top-level Metrics ──────────────────────────────────────────
    st.header("Overall Performance")
    
    col1, col2, col3 = st.columns(3)
    
    def format_metric(val, thresh):
        color = "green" if val >= thresh else "red"
        return f"<h2 style='color: {color}; margin-bottom: 0;'>{val:.3f}</h2><p>Threshold: {thresh}</p>"

    with col1:
        st.markdown("**Faithfulness** *(Grounded reasoning)*")
        st.markdown(format_metric(aggregate.get("faithfulness", 0.0), threshold), unsafe_allow_html=True)
        
    with col2:
        st.markdown("**Answer Relevancy** *(Addresses JD criteria)*")
        st.markdown(format_metric(aggregate.get("answer_relevancy", 0.0), threshold), unsafe_allow_html=True)
        
    with col3:
        st.markdown("**Context Precision** *(Top chunks relevant)*")
        st.markdown(format_metric(aggregate.get("context_precision", 0.0), threshold), unsafe_allow_html=True)

    st.divider()

    # ── Detailed Results Table ─────────────────────────────────────
    st.header("Detailed Case Analysis")

    if details:
        df = pd.DataFrame(details)
        
        # Display only relevant columns to avoid clutter
        display_cols = ["question", "faithfulness", "answer_relevancy", "context_precision"]
        
        # Filter available columns
        available_cols = [c for c in display_cols if c in df.columns]
        
        # Style the dataframe
        def highlight_failing(val):
            if isinstance(val, float):
                color = '#ffcccc' if val < threshold else '#ccffcc'
                return f'background-color: {color}'
            return ''

        styled_df = df[available_cols].style.map(
            highlight_failing, 
            subset=[c for c in available_cols if c != "question"]
        ).format({
            "faithfulness": "{:.3f}",
            "answer_relevancy": "{:.3f}",
            "context_precision": "{:.3f}",
        })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Row selection for deep dive
        st.subheader("Row Inspector")
        selected_index = st.selectbox("Select a test case to inspect details:", df.index)
        
        if selected_index is not None:
            row = df.iloc[selected_index]
            
            st.markdown("### Job Description / Criteria")
            st.info(row.get("question", "N/A"))
            
            st.markdown("### Ground Truth Expectation")
            st.success(row.get("ground_truth", "N/A"))
            
            st.markdown("### Scorer's Answer")
            st.warning(row.get("answer", "N/A"))
            
            st.markdown("### Retrieved Contexts")
            contexts = row.get("contexts", [])
            for i, ctx in enumerate(contexts):
                with st.expander(f"Chunk {i+1}"):
                    st.write(ctx)
    else:
        st.info("No detailed case data available.")


if __name__ == "__main__":
    main()
