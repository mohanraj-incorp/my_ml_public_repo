# Architecture Decision Log

This document records the key design decisions made for the IMDB Movie Agent.
Each entry includes the decision, alternatives considered, and rationale.
This is the interview companion document — reference it when asked "why did you choose X?"

---

## ADR-001: Orchestrator Framework — LangGraph StateGraph

**Decision:** Use LangGraph `StateGraph` for the orchestrator, not Google ADK or raw LangChain.

**Alternatives considered:**
- Google ADK: Full-featured agent platform, but requires GCP credentials and adds operational complexity for a local demo
- LangChain `AgentExecutor`: Higher-level but less explicit about control flow; harder to whiteboard
- LangGraph `create_react_agent` at orchestrator level: Over-powered; the orchestrator's job is routing, not tool-calling

**Rationale:**
1. **Whiteboard-explainable**: StateGraph is a directed graph — nodes and edges can be drawn on a whiteboard. Every possible conversation path is declared explicitly
2. **LLM-agnostic**: Works with any LangChain-compatible LLM (OpenAI, Anthropic, local Ollama)
3. **Built-in memory**: MemorySaver checkpointer handles session state without custom code
4. **Explicit over implicit**: Routes are declared as edges, not emergent from agent reasoning — easier to audit and debug

---

## ADR-002: Multi-Agent Split — 3 Agents

**Decision:** Three agents: OrchestratorAgent (StateGraph) + AnalyticalAgent (ReAct) + SemanticAgent (ReAct).

**Alternatives considered:**
- Single mega-agent with all tools: Simple but poor separation of concerns; harder to optimise prompts for different query types
- Four agents (separate clarification agent): Added complexity without benefit — clarification is a routing edge, not a separate reasoning process
- Five+ agents (genre agent, director agent, etc.): Over-engineered; the semantic agent handles all thematic queries

**Rationale:**
- **Orchestrator as router** (StateGraph): Deterministic routing is more auditable than ReAct reasoning for a routing task
- **Analytical as ReAct**: Query → schema → SQL → result sequences benefit from ReAct's flexible multi-step tool use
- **Semantic as ReAct**: Search → summarise sequences may vary; ReAct handles the uncertainty
- **Clarification as graph edge**: Handled in `route_after_classify()` conditional edge, not a separate agent

---

## ADR-003: RAG Pipeline — Hybrid BM25 + FAISS with Cross-Encoder Reranking

**Decision:** Two-stage pipeline: BM25 + FAISS → RRF fusion → cross-encoder reranking.

**Alternatives considered:**
- BM25 only: Fast and explainable, but misses semantic similarity (fails for "movies about grief")
- Vector only: Good semantics, poor at exact keyword matching (names, years, rare terms)
- Dense retrieval with ColBERT: Better than bi-encoder but 10x slower, overkill for 1000 docs
- Cohere Rerank API: Better accuracy than local cross-encoder, but requires API key + $

**Rationale:**
- **BM25 (rank-bm25)**: Catches exact keyword matches that vectors miss
- **FAISS (IndexFlatL2)**: Exact nearest-neighbour search; fast enough for 1000 docs; no approximation loss
- **RRF (k=60)**: No hyperparameters to tune; consistently outperforms linear score combination
- **Cross-encoder**: Joint (query, passage) encoding for reranking; more accurate than bi-encoder; free + local

**Funnel:** 50 BM25 + 50 FAISS → RRF dedup → top-20 → cross-encoder → top-10

---

## ADR-004: Embeddings — sentence-transformers all-MiniLM-L6-v2

**Decision:** Local sentence-transformers model, not OpenAI text-embedding-3-small.

**Alternatives considered:**
- OpenAI text-embedding-3-small: Higher quality, but requires API call for every search (latency + cost)
- BAAI/bge-m3: Better multilingual, but overkill for English-only IMDB data and heavier to load

**Rationale:**
- Free, runs on CPU, ~80MB download — no extra API key
- Quality is sufficient for 1000 English movie plot descriptions
- Indexes persist to disk — embeddings are computed once, not per query

---

## ADR-005: Data Layer — SQLite only

**Decision:** Single data layer: SQLite for all analytical queries (filtering, aggregation, ranking, exact lookups).

**Alternatives considered:**
- pandas only: Handles 1000 rows fine but doesn't scale (GROUP BY loads full dataset into memory); SQL is more universally understood
- pandas + SQLite (original design): Added a boundary rule ("use pandas for X, SQLite for Y") that was unnecessary complexity for no functional gain at this scale

**Rationale:**
- SQL is universally understood — easier to explain, audit, and hand off than pandas method chains
- SQLite indexes on Director, Genre, Released_Year, IMDB_Rating make WHERE and GROUP BY fast
- Single tool (sql_query) is simpler for the LLM agent to reason about than two tools with a subtle boundary rule
- pandas is still used *internally* in two places that are not agent tools:
    - `init_sqlite_db()`: pd.read_csv() + df.to_sql() to seed the DB once at startup
    - `run_sqlite_query()`: pd.DataFrame(rows).to_string() for result table formatting
- **Scaling path**: swap `sqlite3.connect()` for `duckdb.connect()` or `psycopg2.connect()` — zero interface change for the agent

---

## ADR-006: Idempotent Tool Caching — In-Memory SHA-256 Hash

**Decision:** In-memory dict with TTL, keyed by SHA-256 hash of (function_name, args).

**Alternatives considered:**
- SQLite-backed cache: Persistent across restarts but adds I/O per cache check
- Redis: Distributed, production-grade, but over-engineered for local demo
- No caching: ReAct agents frequently call the same tool with identical args during replanning

**Rationale:**
- SHA-256 of JSON-serialised args is deterministic, collision-resistant, and explainable
- `sort_keys=True` ensures arg dict ordering doesn't affect the hash
- `default=str` makes non-serialisable types (DataFrames) hashable without crashing
- **Production upgrade path**: Replace `_CACHE` dict with `redis.Redis()` — decorator interface unchanged

---

## ADR-007: Retry Strategy — tenacity with Exponential Backoff

**Decision:** Apply tenacity retry decorator only to the LLM routing call in the orchestrator.

**Alternatives considered:**
- Retry everywhere: Adds complexity; most tool calls fail due to logic errors, not transience
- No retries: OpenAI rate limits and network blips cause intermittent failures in demos

**Rationale:**
- The routing LLM call is the single most critical I/O operation (every query goes through it)
- 3 retries with exponential backoff (max 30s) handles transient OpenAI errors without long waits
- Tool calls fail fast and surface errors to the agent (which can replan)

---

## ADR-008: Memory Architecture

**Short-term (in-session):** LangGraph MemorySaver
- Checkpoints full AgentState per thread_id (= session_id)
- Persists across graph invocations within a session
- **Production upgrade:** `SqliteSaver` or `PostgresSaver` — same interface

**Long-term (cross-session):** SQLite `user_preferences` table
- Stores user-expressed preferences as JSON dict per session
- Read at session start, injected into routing context
- Extracted asynchronously after each response (zero latency impact)

---

## ADR-009: Guardrails Placement

**Input guardrails (input_guards.py):**
- Length check → injection detection → topic relevance (cheapest first)
- Only the topic relevance check uses an LLM call

**Tool guardrails (tool_guards.py):**
- SQL write-operation detection (defence-in-depth; primary safety is parameterised queries)
- Result size cap (prevents context window overflow)
- Timeout wrapper (prevents hanging tool calls)

**Output guardrails (output_guards.py):**
- PII regex redaction
- Length cap with truncation notice
- Hallucination check (LLM-based, skipped for SQL/analytical answers)

---

## ADR-010: Async Strategy — Strategic, Not Blanket

**Async used where it materially matters:**
1. `agents/orchestrator.py`: All nodes are `async def` — LangGraph requires this
2. `rag/retriever.py`: BM25 + FAISS run concurrently via `asyncio.gather()` — saves ~10–50ms per query
3. `memory/long_term.py`: `aiosqlite` for preference reads/writes — non-blocking I/O
4. `agents/analytical_agent.py` + `agents/semantic_agent.py`: `ainvoke()` for LLM calls — non-blocking

**Sync kept where async adds complexity without benefit:**
- `tools/pandas_tools.py`: CPU-bound; `asyncio.to_thread()` adds overhead without parallelism
- `tools/sqlite_tools.py`: Fast local queries; async wrapper buys nothing
- `logging_/callbacks.py`: Sync file writes are fine for append-only JSONL logs

---

## ADR-011: Evaluation Framework

**RAGAS metrics:**
- `faithfulness`: Are cited facts supported by retrieved context? (hallucination detector)
- `answer_relevancy`: Does the answer address the question?
- `context_recall`: Does context cover the expected answer?
- `context_precision`: Is retrieved context actually relevant?

**Custom metrics:**
- `structured_accuracy`: Exact fact matching for Type A queries (years, ratings, names)
- `entity_coverage`: Movie title coverage for Type B/C semantic answers
- `clarification_rate`: Type D ambiguous query handling rate

**Golden dataset:** 20 hand-labelled Q&A pairs covering all four query types (A/B/C/D).
Baseline metrics committed to `evaluation/baseline_metrics.json` for regression detection.

---

## Production Upgrade Path Summary

| Component | Current (demo) | Production |
|---|---|---|
| Orchestrator memory | `MemorySaver` (in-process) | `PostgresSaver` |
| Long-term memory | SQLite | PostgreSQL with user auth |
| Tool cache | In-memory dict | Redis |
| Vector store | FAISS flat | FAISS IVF or Pinecone |
| Embeddings | sentence-transformers CPU | GPU or OpenAI API |
| Reranker | local cross-encoder | Cohere Rerank API |
| Analytics DB | SQLite | DuckDB or BigQuery |
| Observability | JSONL files | OpenTelemetry + LangSmith |
| Topic classifier | LLM call | Fine-tuned DistilBERT |
| Config | `.env` | AWS Secrets Manager |
