"""
Central configuration via Pydantic BaseSettings v2.
All secrets come from .env — no hardcoded values anywhere.

pydantic-settings v2: field names auto-map to env vars (uppercase).
  openai_api_key → OPENAI_API_KEY
  llm_model      → LLM_MODEL
  etc.

SCALE NOTE: In production, override the settings source to pull from
AWS Secrets Manager or HashiCorp Vault by implementing
settings_customise_sources(). The rest of the codebase never changes.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # pydantic-settings v2: field names are uppercased to find env vars
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    # Default "" so the module can be imported without a .env file (useful for tests).
    # The actual API call will fail with a clear auth error if the key is empty.
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # ── Limits (retries / recursion / tokens) ─────────────────────────────────
    max_tokens_per_response: int = 1000
    max_agent_iterations: int = 10   # LangGraph recursion_limit
    llm_retry_attempts: int = 3
    llm_retry_max_wait: int = 30     # seconds

    # ── RAG pipeline ──────────────────────────────────────────────────────────
    bm25_top_k: int = 50             # candidates from sparse retrieval
    vector_top_k: int = 50           # candidates from dense retrieval
    rrf_k: int = 60                  # RRF constant (standard = 60)
    rerank_top_n: int = 10           # final results after reranking
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Data paths ────────────────────────────────────────────────────────────
    csv_path: str = "data/imdb_movies.csv"
    db_path: str = "data/imdb.db"
    faiss_index_path: str = "data/faiss_index"
    bm25_index_path: str = "data/bm25_index.pkl"

    # ── Memory ────────────────────────────────────────────────────────────────
    session_memory_window: int = 10

    # ── Voice (optional feature, off by default) ──────────────────────────────
    enable_voice: bool = False
    tts_model: str = "tts-1"
    stt_model: str = "whisper-1"

    # ── Guardrails ────────────────────────────────────────────────────────────
    max_input_chars: int = 500
    max_output_chars: int = 2000
    max_sql_rows: int = 50
    tool_timeout_seconds: int = 10

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir: str = "logs/traces"


# Singleton — import `settings` everywhere instead of re-instantiating.
settings = Settings()
