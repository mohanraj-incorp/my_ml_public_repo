"""
Tool-level guardrails: protect against unsafe or runaway tool invocations.

Three concerns addressed here:
  1. SQL write-operation detection — prevent accidental data modification
  2. Result size cap — stop the LLM context window from being flooded
  3. Execution timeout — prevent tools from blocking indefinitely

WHY TOOL GUARDRAILS: The ReAct agent can call tools with arbitrary inputs.
Without guardrails, a bad SQL query could return all 1000 rows into the LLM
context, causing token-limit errors or cost spikes.

NOTE: SQL injection prevention (the #1 concern) is handled at the tool level
by using parameterised queries (? placeholders). These guardrails are
defence-in-depth on top of that — they catch logical errors, not injection.

SCALE NOTE: For multi-tenant production, inject a mandatory
  WHERE user_id = ? AND tenant_id = ?
clause into every SQL query for row-level security.
PostgreSQL supports this natively via Row Level Security policies.
"""
import asyncio
import logging
import re

from config.settings import settings

logger = logging.getLogger(__name__)

# Only SELECT is allowed — catches accidental DROP/DELETE/INSERT from the LLM
_WRITE_OPS_RE = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


class ToolGuardError(Exception):
    """Raised when a tool call violates a guardrail. Message is logged and surfaced to agent."""
    pass


def validate_sql(query: str) -> None:
    """
    Reject SQL queries that contain write operations or schema enumeration.

    This is defence-in-depth. Primary SQL safety is parameterised queries.
    A production system would use a full SQL parser (e.g., sqlglot) here.
    """
    if _WRITE_OPS_RE.search(query):
        raise ToolGuardError(f"SQL guardrail: write operations not allowed. Query: {query[:100]}")

    query_lower = query.lower()
    if "sqlite_master" in query_lower or "sqlite_schema" in query_lower:
        raise ToolGuardError("SQL guardrail: schema enumeration queries not allowed.")


def cap_result_size(results: list, max_rows: int = None) -> list:
    """
    Truncate result lists before they reach the LLM context.
    Logs a warning so ops teams can see when queries return too many rows.
    """
    max_rows = max_rows or settings.max_sql_rows
    if len(results) > max_rows:
        logger.warning(f"Result capped: {len(results)} → {max_rows} rows")
        return results[:max_rows]
    return results


async def with_timeout(coroutine, timeout_seconds: int = None):
    """
    Run an async coroutine with a hard timeout.

    Used for tool calls that could hang (slow queries, network calls).
    Raises ToolGuardError instead of blocking indefinitely.

    Args:
        coroutine:       The awaitable to run
        timeout_seconds: Max wait time. Defaults to settings.tool_timeout_seconds.
    """
    timeout_seconds = timeout_seconds or settings.tool_timeout_seconds
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise ToolGuardError(
            f"Tool call timed out after {timeout_seconds}s. Try a more specific query."
        )
