# 🧠 Resume Screener — Multi-Agent AI Pipeline

A production-grade, multi-agent Resume Screener system built with **LangChain**, **LangGraph**, and **GPT-4o**. Uses RAG + semantic similarity for candidate matching, structured LLM output for scoring, and an independent bias auditing agent.

---

## Architecture

```
┌─────────────┐     ┌────────────┐     ┌───────────────┐
│  Screener   │────▶│   Scorer   │────▶│ Bias Auditor  │
│  (RAG+Top-K)│     │(Structured)│     │ (Independent) │
└─────────────┘     └────────────┘     └───────┬───────┘
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              Release Results      Human Review
                            (risk=low/medium)      (risk=high)
```

### Three-Agent Pipeline

| Agent | Role | Key Design Choice |
|-------|------|-------------------|
| **Screener** | RAG retrieval → top-K shortlist | ChromaDB + text-embedding-3-small |
| **Scorer** | Structured LLM scoring per criterion | Pydantic v2 models with evidence validation |
| **Bias Auditor** | Independent fairness review | Receives ONLY reasoning strings — no scores, IDs, or metadata |

## Tech Stack

- **Backend**: Python 3.11, FastAPI, pydantic v2
- **Agents**: LangChain + LangGraph (StateGraph)
- **LLM**: GPT-4o with structured output
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: ChromaDB (self-hosted)
- **NLP**: spaCy en_core_web_trf
- **Evaluation**: RAGAS (faithfulness, answer_relevancy, context_precision)
- **Frontend**: React 18 + Vite + Tailwind CSS + TypeScript
- **Dashboard**: Streamlit (eval metrics)

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 2. Run with Docker Compose
docker compose up --build

# 3. Access services
# API:       http://localhost:8080
# Dashboard: http://localhost:8501
```

## Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_trf

# Run tests
pytest tests/ -v

# Run linter
ruff check .
```

## Project Structure

```
resume-screener/
├── agents/              # Three-agent pipeline
│   ├── screener.py      # RAG retrieval, top-K shortlist
│   ├── scorer.py        # Structured LLM scoring + pydantic models
│   └── bias_auditor.py  # Independent bias audit
├── graph.py             # LangGraph StateGraph wiring
├── preprocessing/       # Document processing pipeline
│   ├── parser.py        # PDF + DOCX extraction
│   ├── anonymizer.py    # spaCy NER: strip PII
│   └── chunker.py       # Section-aware chunking
├── evaluation/          # RAGAS evaluation layer
│   ├── metrics.py       # Metric computation
│   ├── run_eval.py      # CLI eval runner (CI gate)
│   ├── dashboard.py     # Streamlit dashboard
│   └── test_data/       # Synthetic test triples
├── api/                 # FastAPI REST API
│   └── main.py
├── frontend/            # React + Vite + Tailwind
├── tests/               # pytest test suite
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## License

MIT
