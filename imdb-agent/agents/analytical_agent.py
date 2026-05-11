"""
Analytical Agent — answers structured, data-driven movie queries via SQL.

Uses LangGraph's create_react_agent (ReAct = Reason + Act loop) with two tools:
  sql_query          — runs any SELECT against the movies table
  get_column_schema  — inspect column names/types before writing a query

WHY SQLITE ONLY (no pandas tool):
  SQL is the right abstraction for structured data queries. A single tool
  with a rich docstring (examples of GROUP BY, filtering, window functions)
  is simpler to explain and easier for the LLM to use correctly than
  two tools with a subtle boundary rule between them.

  Scaling path: swap sqlite3.connect() for duckdb.connect() or
  psycopg2.connect() — the agent's tool interface is unchanged.

WHY create_react_agent (not a custom StateGraph here):
  ReAct is right for tool-calling agents with uncertain tool sequences.
  The agent may need: get_schema → sql_query → (refine query) → sql_query.
  create_react_agent handles this multi-step reasoning loop out-of-the-box.

COMPARE TO ORCHESTRATOR: Orchestrator is a StateGraph (deterministic routing).
  This agent is ReAct (flexible tool use). Right tool for each job.
"""
import logging
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config.prompts import ANALYTICAL_SYSTEM
from tools.sqlite_tools import run_sqlite_query, get_schema

logger = logging.getLogger(__name__)


# ── Tool definitions ────────────────────────────────────────────────────────────

@tool
def sql_query(query: str) -> str:
    """
    Run a SQL SELECT query against the IMDB movies database.

    Use for ALL structured queries: exact lookups, filtering, aggregation,
    ranking, GROUP BY, and window functions.

    Table: 'movies'. Call get_column_schema first if unsure about column names.

    Args:
        query: SQL SELECT statement (read-only). Examples:

               -- Exact fact lookup
               SELECT Series_Title, Released_Year, IMDB_Rating
               FROM movies WHERE Series_Title LIKE '%Dark Knight%'

               -- Aggregation
               SELECT Director, COUNT(*) AS films, ROUND(AVG(IMDB_Rating),2) AS avg_rating
               FROM movies GROUP BY Director HAVING COUNT(*) >= 2
               ORDER BY avg_rating DESC LIMIT 10

               -- Decade filter
               SELECT COUNT(*) AS total FROM movies
               WHERE Released_Year BETWEEN 1990 AND 1999

               -- Multi-condition filter
               SELECT Series_Title, IMDB_Rating, Meta_score FROM movies
               WHERE IMDB_Rating > 8.5 AND Meta_score > 80
               ORDER BY IMDB_Rating DESC
    """
    return run_sqlite_query(query)


@tool
def get_column_schema() -> str:
    """
    Return the movies table schema: column names, data types, and a sample row.

    Call this first when unsure about column names, data types, or value formats
    before writing a SQL query. Prevents hallucinating wrong column names.
    """
    return get_schema()


# ── Agent factory ───────────────────────────────────────────────────────────────

def create_analytical_agent(llm: ChatOpenAI):
    """
    Build and return the analytical ReAct agent.

    Args:
        llm: Configured ChatOpenAI instance (with callbacks already attached
             by the orchestrator — do not create a new one here)

    Returns:
        Compiled LangGraph ReAct agent.
        Call with: agent.ainvoke({"messages": [HumanMessage(content=query)]})
    """
    tools = [sql_query, get_column_schema]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=ANALYTICAL_SYSTEM,
    )

    logger.info("Analytical agent ready — tools: sql_query, get_column_schema")
    return agent
