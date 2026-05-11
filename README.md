# IMDB Movie Agent

Conversational voice agent over the IMDB top-1000 movies dataset.
Built with LangGraph, hybrid RAG (BM25 + FAISS), and Streamlit.

## Quick Start

```bash
# 1. Clone and set up environment
cd imdb-agent
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY

# 4. Download the dataset
# Place imdb_movies.csv in data/imdb_movies.csv
# Dataset: https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows

# 5. Build search indexes (run once)
python scripts/build_indexes.py

# 6. Launch the app
streamlit run app/main.py
```

## Architecture Overview

```
User Query
    │
    ▼
[Input Guardrails]  ← length, injection, topic relevance
    │
    ▼
[Orchestrator]      ← LangGraph StateGraph
    │
    ├─ analytical ──► [Analytical Agent]  ← SQLite + pandas tools
    ├─ semantic   ──► [Semantic Agent]    ← BM25 + FAISS + reranker
    └─ clarify    ──► Ask clarification question → wait for user
    │
    ▼
[Output Guardrails] ← PII, length cap, hallucination check
    │
    ▼
User Response
```

For full design decisions, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Query Types Supported

| Type | Example | Handler |
|------|---------|---------|
| **A** Structured | "When did The Matrix release?" | Analytical Agent → SQL |
| **B** Semantic | "Comedy movies with death in the plot" | Semantic Agent → RAG |
| **C** Hybrid | "Summarize Spielberg's sci-fi movies" | Semantic Agent → Search + Summarize |
| **D** Ambiguous | "Al Pacino movies over $50M" | Orchestrator → Clarification |

## Running Evaluation

```bash
python evaluation/run_eval.py
```

Runs RAGAS metrics (faithfulness, relevancy, recall, precision) + custom metrics
(structured accuracy, entity coverage, clarification rate) over 20 golden examples.

## Enabling Voice

```bash
# In .env:
ENABLE_VOICE=true
```

Then toggle "Voice Mode" in the Streamlit sidebar. Requires OpenAI API credits.
Uses Whisper (STT) and TTS-1 (text-to-speech).

## Project Structure

```
imdb-agent/
├── agents/          # LangGraph orchestrator + ReAct sub-agents
├── app/             # Streamlit UI + voice wrappers
├── config/          # Settings (Pydantic) + all prompts
├── data/            # CSV, SQLite DB, FAISS index (generated)
├── evaluation/      # Golden dataset + RAGAS + custom metrics
├── guardrails/      # Input / output / tool-level guardrails
├── logging_/        # Callback handler + log schemas
├── memory/          # Short-term (MemorySaver) + long-term (SQLite)
├── rag/             # Indexer, hybrid retriever, cross-encoder reranker
├── scripts/         # build_indexes.py, seed_golden_dataset.py
├── tests/           # Unit + integration tests
└── tools/           # pandas, SQLite, search, cache, summarize tools
```
