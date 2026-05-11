"""
Structured log entry schema for agent and tool traceability.

Using dataclasses for lightweight, JSON-serialisable records.
Every tool call, agent decision, and guardrail check gets its own log line.

SCALE NOTE: In production, replace dataclass serialisation with OpenTelemetry
spans (pip install opentelemetry-sdk) to get distributed tracing across
microservices with zero code changes in business logic.
"""
import uuid
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


def _now() -> str:
    return datetime.utcnow().isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


@dataclass
class ToolCallLog:
    """One record per tool invocation."""
    session_id: str
    trace_id: str = field(default_factory=_uid)
    timestamp: str = field(default_factory=_now)
    tool_name: str = ""
    input_summary: str = ""     # Truncated input for readability
    output_summary: str = ""    # Truncated output — full result stays in agent state
    latency_ms: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class AgentCallLog:
    """One record per orchestrator routing decision or sub-agent invocation."""
    session_id: str
    trace_id: str = field(default_factory=_uid)
    timestamp: str = field(default_factory=_now)
    agent_name: str = ""
    user_query: str = ""
    routing_decision: str = ""   # "analytical" | "semantic" | "clarify" | "blocked"
    response_summary: str = ""   # First 200 chars of final response
    total_latency_ms: float = 0.0
    tool_calls_made: list = field(default_factory=list)
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class GuardrailLog:
    """One record per guardrail check (pass or fail)."""
    session_id: str
    trace_id: str = field(default_factory=_uid)
    timestamp: str = field(default_factory=_now)
    guardrail_type: str = ""   # "input" | "output" | "tool"
    check_name: str = ""
    passed: bool = True
    details: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))
