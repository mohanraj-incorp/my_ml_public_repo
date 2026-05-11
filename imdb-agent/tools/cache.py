"""
Idempotent tool call cache using SHA-256 content hashing.

WHY: LLM agents (especially ReAct) may call the same tool with identical
arguments multiple times in one conversation — during retries, re-planning,
or redundant reasoning steps. Caching eliminates duplicate computation.

DESIGN: In-memory dict with TTL. Simple to explain, zero dependencies.

SCALE NOTE: For multi-worker production, swap the in-memory dict for Redis:
    import redis
    _cache = redis.Redis(host="localhost", decode_responses=True)
    _cache.setex(key, ttl_seconds, json.dumps(result))
The decorator interface stays identical — only the storage backend changes.
"""
import hashlib
import json
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# Process-global cache dict. In prod: replace with Redis client.
_CACHE: dict[str, dict] = {}


def cached_tool(ttl_seconds: int = 3600):
    """
    Decorator that caches tool results by a hash of (function_name, args, kwargs).

    Args:
        ttl_seconds: Cache TTL in seconds. Pass 0 to disable (useful in tests).

    Usage:
        @cached_tool(ttl_seconds=3600)
        def run_sqlite_query(sql: str) -> str:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if ttl_seconds == 0:
                return func(*args, **kwargs)

            # Build a deterministic key: function name + all arguments.
            # sort_keys=True: {"a":1,"b":2} and {"b":2,"a":1} produce the same hash.
            # default=str: handles non-serialisable args (DataFrames etc.) without crashing.
            key_payload = json.dumps(
                {"fn": func.__name__, "args": list(args), "kwargs": kwargs},
                sort_keys=True,
                default=str,
            )
            cache_key = hashlib.sha256(key_payload.encode()).hexdigest()

            # Cache HIT — return early if within TTL
            if cache_key in _CACHE:
                entry = _CACHE[cache_key]
                age = time.time() - entry["ts"]
                if age < ttl_seconds:
                    logger.debug(f"[cache HIT]  {func.__name__} (age={age:.1f}s)")
                    return entry["result"]
                logger.debug(f"[cache STALE] {func.__name__} (age={age:.1f}s)")

            # Cache MISS — execute and store
            result = func(*args, **kwargs)
            _CACHE[cache_key] = {"result": result, "ts": time.time()}
            logger.debug(f"[cache MISS]  {func.__name__} → stored key={cache_key[:8]}…")
            return result

        return wrapper
    return decorator


def clear_cache() -> None:
    """Clear all cached entries. Call between evaluation runs for clean results."""
    _CACHE.clear()


def cache_stats() -> dict:
    """Return basic stats for monitoring / debugging."""
    now = time.time()
    entries = list(_CACHE.values())
    return {
        "total_entries": len(entries),
        "oldest_age_seconds": max((now - e["ts"]) for e in entries) if entries else 0,
    }
