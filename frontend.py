"""Frontend Dashboard — Streamlit UI for the Resume Screener.

Interacts with the FastAPI backend (main.py) to:
  1. Upload resumes for ingestion
  2. Enter a job description and run the screening pipeline
  3. Visualize scoring results and bias audit reports

Usage:
    streamlit run frontend.py
"""

import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Resume Screener | Dashboard",
    page_icon="📄",
    layout="wide",
)

API_URL = "http://localhost:8000"

st.title("📄 Multi-Agent Resume Screener")
st.markdown("Fair, structured, and auditable candidate screening powered by AI.")

tab1, tab2 = st.tabs(["📥 Ingest Resumes", "🎯 Screen Candidates"])

# ── Tab 1: Ingest Resumes ──────────────────────────────────────────

with tab1:
    st.header("Upload Resumes")
    st.markdown("Upload candidate resumes (PDF or DOCX) to be anonymized, chunked, and ingested into the vector store.")

    uploaded_files = st.file_uploader(
        "Choose resume files", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    
    use_ner = st.checkbox("Enable Deep Anonymization (spaCy NER)", value=True)

    if st.button("Ingest Files", type="primary", disabled=not uploaded_files):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            
            try:
                # Send to FastAPI /ingest endpoint
                files = {"file": (file.name, file.getvalue(), file.type)}
                data = {"use_ner": use_ner}
                
                res = requests.post(f"{API_URL}/ingest", files=files, data=data)
                
                if res.status_code == 200:
                    results.append({"file": file.name, "status": "Success", **res.json()})
                else:
                    results.append({"file": file.name, "status": "Failed", "error": res.text})
            except Exception as e:
                results.append({"file": file.name, "status": "Error", "error": str(e)})
                
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("Ingestion Complete!")
        
        # Display results
        if results:
            df = pd.DataFrame(results)
            # Cleanup view: only select columns that exist in the dataframe
            desired_cols = ["file", "status", "candidate_id", "chunks_stored", "anonymization_count", "error"]
            display_cols = [c for c in desired_cols if c in df.columns]
            display_df = df[display_cols].copy()
            st.dataframe(display_df, use_container_width=True)


# ── Tab 2: Screen Candidates ───────────────────────────────────────

with tab2:
    st.header("Run Screening Pipeline")
    st.markdown("Paste a job description to trigger the Multi-Agent LangGraph pipeline (Retrieve → Score → Audit Bias).")

    col1, col2 = st.columns([3, 1])
    
    with col1:
        jd_text = st.text_area("Job Description", height=200, placeholder="Paste JD here...")
        
    with col2:
        top_k = st.number_input("Candidates to Shortlist", min_value=1, max_value=20, value=5)
        
    if st.button("Run Screening", type="primary", disabled=not jd_text.strip()):
        with st.spinner("Pipeline running... This may take a minute as 3 agents do their work."):
            start_time = time.time()
            
            try:
                res = requests.post(
                    f"{API_URL}/screen", 
                    json={"jd_text": jd_text, "top_k": top_k}
                )
                
                if res.status_code != 200:
                    st.error(f"API Error: {res.text}")
                    st.stop()
                    
                data = res.json()
                elapsed = round(time.time() - start_time, 1)
                
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Make sure `uvicorn main:app` is running.")
                st.stop()

        # Display Pipeline Results
        st.success(f"Pipeline finished in {elapsed}s")
        
        status = data.get("status")
        bias_report = data.get("bias_report", {})
        
        # 1. Bias Audit Report
        st.subheader("🕵️ Bias Auditor Report")
        risk = bias_report.get("overall_risk", "unknown").upper()
        
        if risk == "LOW":
            st.success(f"Risk Level: {risk} — {bias_report.get('summary')}")
        elif risk == "MEDIUM":
            st.warning(f"Risk Level: {risk} — {bias_report.get('summary')}")
        else:
            st.error(f"Risk Level: {risk} — {bias_report.get('summary')}")
            
        flags = bias_report.get("flagged_patterns", [])
        if flags:
            with st.expander("View Flagged Bias Patterns"):
                for flag in flags:
                    st.markdown(f"**Pattern:** {flag.get('pattern')} (Frequency: {flag.get('frequency')})")
                    st.markdown(f"*Quote:* \"{flag.get('example_quote')}\"")
                    st.markdown(f"*Recommendation:* {flag.get('recommendation')}")
                    st.divider()

        # 2. Blocked Release
        if status == "human_review_required":
            st.error("🚨 **RELEASE BLOCKED**: The independent Bias Auditor flagged high-risk reasoning patterns. "
                     "Candidate scores will not be displayed until a human reviewer clears them.")
            st.stop()
            
        # 3. Candidate Scores
        scores = data.get("scores", [])
        jd_criteria = data.get("jd_criteria", [])
        
        if not scores:
            st.warning("No candidates were scored. Ensure resumes have been ingested.")
            st.stop()
            
        st.subheader(f"🏆 Top {len(scores)} Candidate Matches")
        
        with st.expander("View Extracted JD Criteria"):
            for c in jd_criteria:
                st.markdown(f"- {c}")
                
        # Sort scores descending
        scores = sorted(scores, key=lambda x: x.get("overall_score", 0), reverse=True)
        
        for i, score in enumerate(scores):
            rec = score.get("recommendation", "").upper()
            rec_color = "green" if rec == "SHORTLIST" else "orange" if rec == "HOLD" else "red"
            
            with st.expander(f"#{i+1} | Candidate {score.get('candidate_id')[:8]}... | Score: {score.get('overall_score')}/10"):
                st.markdown(f"**Recommendation:** :{rec_color}[{rec}] (Confidence: {score.get('confidence')})")
                
                st.markdown("### Criteria Breakdown")
                criteria_scores = score.get("criteria", [])
                
                for crit in criteria_scores:
                    st.markdown(f"**{crit.get('criterion')}** — Score: {crit.get('score')}/10")
                    st.info(f"**Reasoning:** {crit.get('reasoning')}")
                    st.caption(f"*Evidence Quote:* \"{crit.get('evidence')}\"")
                    st.divider()
