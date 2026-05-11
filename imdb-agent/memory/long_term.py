"""
Long-term user preference memory using SQLite + aiosqlite.

Stores preferences that persist across sessions: preferred genres, rating
thresholds, decade preferences, etc. Read at session start and injected into
the orchestrator's routing context so the agents are preference-aware.

WHY SQLITE: Zero infrastructure (file on disk), sufficient for demo.
Async via aiosqlite so preference reads/writes don't block the event loop.

SCALE NOTE: In production with many concurrent users:
  - PostgreSQL for concurrent writes (SQLite serialises all writes)
  - Scope by authenticated user_id, not session_id
  - Add preference TTL — stale preferences from months ago degrade UX
"""
import json
import logging
from datetime import datetime

import aiosqlite

from config.settings import settings
from config.prompts import PREFERENCE_EXTRACT_PROMPT

logger = logging.getLogger(__name__)

DB_PATH = settings.db_path


async def init_preferences_table() -> None:
    """Create user_preferences table if it doesn't exist. Call once at app startup."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                session_id   TEXT PRIMARY KEY,
                preferences  TEXT NOT NULL DEFAULT '{}',
                updated_at   TEXT NOT NULL
            )
        """)
        await db.commit()
    logger.info("user_preferences table ready.")


async def get_preferences(session_id: str) -> dict:
    """
    Fetch stored preferences for a session.
    Returns {} if no preferences have been stored yet.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT preferences FROM user_preferences WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
    return json.loads(row[0]) if row else {}


async def upsert_preferences(session_id: str, updates: dict) -> None:
    """
    Merge new preference values into existing ones and persist.

    WHY MERGE: We accumulate preferences across the conversation.
    A user might say "I like sci-fi" early and "only highly rated movies"
    later — both should be remembered simultaneously.
    """
    existing = await get_preferences(session_id)
    merged = {**existing, **updates}  # later values override earlier for same keys

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_preferences (session_id, preferences, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                preferences = excluded.preferences,
                updated_at  = excluded.updated_at
            """,
            (session_id, json.dumps(merged), datetime.utcnow().isoformat()),
        )
        await db.commit()
    logger.info(f"Preferences updated for {session_id}: {updates}")


async def extract_and_save_preferences(session_id: str, user_message: str, llm) -> None:
    """
    Opportunistically detect and save preferences expressed in a message.

    Runs after the main response is sent (fire-and-forget) so it adds zero
    latency to the user-facing turn.

    HEURISTIC: Only fires when the message contains preference signal words
    to avoid spending LLM tokens on every casual message.

    SCALE NOTE: Replace the LLM call with a fine-tuned classifier for
    lower cost and higher throughput in production.
    """
    signals = ["i like", "i prefer", "i love", "i hate", "i enjoy", "show me", "only", "always"]
    if not any(s in user_message.lower() for s in signals):
        return  # No signal words — skip the LLM call

    prompt = PREFERENCE_EXTRACT_PROMPT.format(message=user_message)
    try:
        response = await llm.ainvoke(prompt)
        prefs = json.loads(response.content.strip())
        if prefs:
            await upsert_preferences(session_id, prefs)
    except Exception as e:
        # Non-critical: preference extraction failure must never break the conversation
        logger.debug(f"Preference extraction skipped: {e}")
