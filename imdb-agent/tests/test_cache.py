"""Unit tests for the idempotent tool cache."""
import time
import pytest
from tools.cache import cached_tool, clear_cache, cache_stats


def test_cache_hit_returns_same_result():
    clear_cache()  # Isolate from other tests sharing the global _CACHE
    call_count = 0

    @cached_tool(ttl_seconds=60)
    def my_tool(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    result1 = my_tool(5)
    result2 = my_tool(5)

    assert result1 == result2 == 10
    assert call_count == 1, "Function should only execute once on cache hit"


def test_different_args_produce_different_cache_entries():
    clear_cache()  # _CACHE is module-global; clear to avoid cross-test hits
    call_count = 0

    @cached_tool(ttl_seconds=60)
    def my_tool(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    my_tool(5)
    my_tool(10)
    assert call_count == 2, "Different args should produce separate cache entries"


def test_ttl_zero_disables_cache():
    call_count = 0

    @cached_tool(ttl_seconds=0)
    def my_tool() -> int:
        nonlocal call_count
        call_count += 1
        return 42

    my_tool()
    my_tool()
    assert call_count == 2, "ttl_seconds=0 should bypass caching"


def test_clear_cache():
    clear_cache()

    @cached_tool(ttl_seconds=60)
    def my_tool() -> str:
        return "result"

    my_tool()
    stats_before = cache_stats()
    clear_cache()
    stats_after = cache_stats()

    assert stats_before["total_entries"] >= 1
    assert stats_after["total_entries"] == 0
