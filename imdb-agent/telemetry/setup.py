"""
Telemetry placeholder — Phoenix (Arize) observability removed.
Observability is handled by the JSONL callback handler in logging_/callbacks.py.
Each session's tool calls, LLM calls, routing decisions, and guardrail events
are logged to logs/traces/<session_id>.jsonl.
"""


def setup_phoenix(project_name: str = "imdb-agent") -> None:
    pass


def flush() -> None:
    pass
