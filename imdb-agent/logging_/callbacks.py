"""
Custom LangChain callback handler for full agent/tool traceability.

Hooks into LangChain's callback system so every LLM call and tool call is
logged automatically — without touching agent code. This is the clean
observability pattern for LangChain/LangGraph apps.

One JSONL file per session. Each line = one self-contained JSON record.
Easy to grep, tail, and audit per user session.

SCALE NOTE: In production, extend on_llm_end / on_tool_end to emit
OpenTelemetry spans or push to LangSmith. The callback interface is the
abstraction boundary — swap backends here, nothing else changes.
"""
import os
import time
import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from logging_.schema import ToolCallLog, AgentCallLog, GuardrailLog
from config.settings import settings

logger = logging.getLogger(__name__)


class IMDBCallbackHandler(BaseCallbackHandler):
    """
    Logs LLM calls, tool calls, and agent actions to a per-session JSONL file.
    Attach to the LLM via callbacks=[IMDBCallbackHandler(session_id)].
    """

    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id
        self._start_times: dict[str, float] = {}  # run_id → wall-clock start

        os.makedirs(settings.log_dir, exist_ok=True)
        self.log_file = os.path.join(settings.log_dir, f"{session_id}.jsonl")

    def _write(self, record: str) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(record + "\n")

    # ── Tool lifecycle ─────────────────────────────────────────────────────────

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **kwargs
    ):
        self._start_times[str(run_id)] = time.time()
        tool_name = serialized.get("name", "unknown_tool")
        logger.info(f"[{self.session_id}] TOOL START  {tool_name} | in={input_str[:80]}")

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs):
        elapsed = (time.time() - self._start_times.pop(str(run_id), time.time())) * 1000
        log = ToolCallLog(
            session_id=self.session_id,
            tool_name=kwargs.get("name", "unknown"),
            output_summary=str(output)[:200],
            latency_ms=round(elapsed, 2),
        )
        self._write(log.to_json())
        logger.info(f"[{self.session_id}] TOOL END    latency={elapsed:.0f}ms")

    def on_tool_error(self, error: Exception, *, run_id: UUID, **kwargs):
        elapsed = (time.time() - self._start_times.pop(str(run_id), time.time())) * 1000
        log = ToolCallLog(
            session_id=self.session_id,
            tool_name=kwargs.get("name", "unknown"),
            latency_ms=round(elapsed, 2),
            error=str(error),
        )
        self._write(log.to_json())
        logger.error(f"[{self.session_id}] TOOL ERROR  {error}")

    # ── LLM lifecycle ──────────────────────────────────────────────────────────

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs
    ):
        self._start_times[f"llm_{run_id}"] = time.time()
        logger.debug(f"[{self.session_id}] LLM START   model={serialized.get('name', '?')}")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        elapsed = (time.time() - self._start_times.pop(f"llm_{run_id}", time.time())) * 1000
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        text = ""
        if response.generations:
            text = getattr(response.generations[0][0], "text", "")[:200]
        log = AgentCallLog(
            session_id=self.session_id,
            agent_name=kwargs.get("name", "llm"),
            response_summary=text,
            total_latency_ms=round(elapsed, 2),
            tool_calls_made=[usage],
        )
        self._write(log.to_json())
        logger.info(
            f"[{self.session_id}] LLM END     latency={elapsed:.0f}ms | tokens={usage}"
        )

    # ── Agent lifecycle ────────────────────────────────────────────────────────

    def on_agent_action(self, action, *, run_id: UUID, **kwargs):
        logger.info(
            f"[{self.session_id}] AGENT ACTION tool={action.tool} | in={str(action.tool_input)[:80]}"
        )

    def on_agent_finish(self, finish, *, run_id: UUID, **kwargs):
        summary = str(finish.return_values)[:200]
        log = AgentCallLog(
            session_id=self.session_id,
            agent_name="react_agent",
            response_summary=summary,
        )
        self._write(log.to_json())
        logger.info(f"[{self.session_id}] AGENT FINISH out={summary[:150]}")

    # ── Guardrail lifecycle ────────────────────────────────────────────────────

    def log_guardrail(self, guardrail_type: str, check_name: str, passed: bool, details: str = "") -> None:
        """Call this directly from guardrail code to persist pass/fail events."""
        log = GuardrailLog(
            session_id=self.session_id,
            guardrail_type=guardrail_type,
            check_name=check_name,
            passed=passed,
            details=details[:200],
        )
        self._write(log.to_json())
        status = "PASS" if passed else "FAIL"
        logger.info(f"[{self.session_id}] GUARDRAIL {status} {guardrail_type}/{check_name}")
