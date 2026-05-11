"""
SQLite tools — the single data layer for all analytical queries.

All structured queries (filtering, aggregation, ranking, exact lookups) go
through SQLite. This is the only tool the Analytical Agent uses.

WHY SQLITE ONLY (no pandas analytical layer):
  - SQL is universally understood — easier to explain and audit than pandas chains
  - SQLite indexes (Director, Genre, Year, Rating) make GROUP BY and WHERE fast
  - The tool interface is identical whether we have 1K or 10M rows
  - Scaling path is a one-line swap: SQLite → DuckDB → PostgreSQL, same SQL

  pandas is still used internally for two things that are NOT agent tools:
    1. init_sqlite_db(): pd.read_csv() + df.to_sql() to seed the DB once
    2. run_sqlite_query(): pd.DataFrame(rows).to_string() for result formatting
  These are implementation details, not the agent-facing data layer.

SCALE NOTE: For analytical workloads >10M rows, replace sqlite3.connect()
with duckdb.connect() — DuckDB speaks the same SQL dialect, adds columnar
storage and vectorised GROUP BY execution. Zero interface change for the agent.
"""
import logging
import os
import sqlite3

import pandas as pd

from config.settings import settings
from guardrails.tool_guards import validate_sql, cap_result_size
from tools.cache import cached_tool

logger = logging.getLogger(__name__)


def init_sqlite_db(csv_path: str = None, db_path: str = None) -> None:
    """
    Load the IMDB CSV into SQLite and create query indexes.
    Call once at app startup. Skips if DB already exists.

    Uses pandas only for CSV parsing and bulk insert (df.to_sql) — this is a
    one-time setup operation, not part of the agent query path.

    WHY DROP/RECREATE on first run: simpler than schema migrations for a demo.
    In production, use Alembic (SQLAlchemy) or Flyway for schema versioning.
    """
    csv_path = csv_path or settings.csv_path
    db_path = db_path or settings.db_path

    if os.path.exists(db_path):
        logger.info(f"SQLite DB exists at {db_path} — skipping init.")
        return

    logger.info(f"Loading {csv_path} → SQLite {db_path}…")
    df = pd.read_csv(csv_path)

    # Normalise column names for SQL friendliness (no spaces, no hyphens)
    df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]
    df["Gross"] = (
        df["Gross"].astype(str).str.replace(",", "").str.replace("$", "")
    )
    df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")
    df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")

    # Drop rows with no plot text and reset index — MUST mirror what indexer.py
    # does so that SQLite movie_id == FAISS/BM25 doc id for every row.
    df = df.dropna(subset=["Overview"]).reset_index(drop=True)

    conn = sqlite3.connect(db_path)
    # index=True writes the DataFrame row number (0, 1, 2…) as column "movie_id".
    # This is the shared primary key that links SQLite rows to FAISS/BM25 docs.
    df.to_sql("movies", conn, if_exists="replace", index=True, index_label="movie_id")

    # Index the most-queried columns — WHERE and GROUP BY benefit significantly
    for idx_sql in [
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_pk      ON movies(movie_id)",
        "CREATE INDEX IF NOT EXISTS idx_director ON movies(Director)",
        "CREATE INDEX IF NOT EXISTS idx_genre    ON movies(Genre)",
        "CREATE INDEX IF NOT EXISTS idx_year     ON movies(Released_Year)",
        "CREATE INDEX IF NOT EXISTS idx_rating   ON movies(IMDB_Rating)",
    ]:
        conn.execute(idx_sql)

    conn.commit()
    conn.close()
    logger.info(f"SQLite DB ready: {len(df)} movies, indexes created.")


def get_schema() -> str:
    """
    Return the movies table schema (column names + types) and one sample row.

    The agent calls this when unsure about column names or data types before
    writing a query. Sourced directly from SQLite's PRAGMA — always accurate.

    Not cached: always reflects the current DB state.
    """
    try:
        conn = sqlite3.connect(settings.db_path)
        cursor = conn.cursor()

        # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
        cursor.execute("PRAGMA table_info(movies)")
        columns = cursor.fetchall()

        # One sample row to show data format
        cursor.execute("SELECT * FROM movies LIMIT 1")
        sample_row = cursor.fetchone()
        col_names = [c[1] for c in columns]
        conn.close()

        lines = ["Table: movies\n", "Columns (name → SQLite type):",
             "  movie_id (INTEGER, primary key — shared with FAISS/BM25 doc ids)"]
        for col in columns:
            lines.append(f"  {col[1]:25s}  {col[2]}")

        if sample_row:
            lines.append("\nSample row:")
            for name, val in zip(col_names, sample_row):
                lines.append(f"  {name:25s}  {repr(val)[:60]}")

        return "\n".join(lines)

    except sqlite3.Error as e:
        return f"Schema unavailable: {e} — ensure init_sqlite_db() has been called."


@cached_tool(ttl_seconds=3600)
def run_sqlite_query(query: str) -> str:
    """
    Execute a read-only SQL SELECT against the IMDB movies database.

    Call get_schema() first if unsure about column names or data types.

    Args:
        query: SQL SELECT statement. Table name: 'movies'

    Returns:
        Formatted results table as a string, or an error message.

    Common patterns:
        -- Exact lookup
        SELECT * FROM movies WHERE Series_Title LIKE '%Matrix%'

        -- Top N by metric
        SELECT Series_Title, IMDB_Rating FROM movies
        ORDER BY IMDB_Rating DESC LIMIT 10

        -- Aggregation with filter
        SELECT Director, ROUND(AVG(IMDB_Rating), 2) AS avg_rating, COUNT(*) AS films
        FROM movies
        GROUP BY Director
        HAVING COUNT(*) >= 2
        ORDER BY avg_rating DESC LIMIT 10

        -- Decade filter
        SELECT COUNT(*) FROM movies
        WHERE Released_Year BETWEEN 1990 AND 1999

        -- Multi-column filter
        SELECT Series_Title, IMDB_Rating, Meta_score
        FROM movies
        WHERE IMDB_Rating > 8.5 AND Meta_score > 80
        ORDER BY IMDB_Rating DESC
    """
    validate_sql(query)  # Reject write operations before touching the DB

    try:
        conn = sqlite3.connect(settings.db_path)
        conn.row_factory = sqlite3.Row  # dict-like rows
        cursor = conn.cursor()
        cursor.execute(query)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e} | query={query[:200]}")
        return f"Database error: {e}"

    if not rows:
        return "Query returned no results."

    rows = cap_result_size(rows, max_rows=settings.max_sql_rows)
    # pandas used only for pretty-printing the result table — not as a data layer
    return pd.DataFrame(rows).to_string(index=False)
