"""
DEPRECATED — not used by any agent.

All analytical queries now go through tools/sqlite_tools.py.

Rationale: SQLite is a cleaner single layer — universally understood SQL,
indexes for performance, and a straight upgrade path (SQLite → DuckDB →
PostgreSQL) without changing any agent code.

pandas is still used as an implementation detail inside sqlite_tools.py
(CSV loading, result formatting) but is no longer an agent-facing tool.
"""
