"""
Short-term (in-session) conversation memory via LangGraph MemorySaver.

MemorySaver stores the full LangGraph state (including message history) in
process memory, keyed by thread_id (= our session_id). Each thread is
fully isolated — one user's history never bleeds into another's.

WHY MEMORYSAVER: It integrates directly with LangGraph's compile() step and
create_react_agent — no custom state serialisation needed. The checkpointer
pattern is the idiomatic LangGraph way to add memory.

WINDOW: We set recursion_limit (not a message trim here — LangGraph MemorySaver
keeps full history by default). For a real window trim, wrap with
langchain_core.messages.trim_messages in the graph node.

SCALE NOTE: MemorySaver is in-process and lost on restart. For persistence:
  from langgraph.checkpoint.sqlite import SqliteSaver
  memory = SqliteSaver.from_conn_string("checkpoints.db")
For distributed multi-server:
  from langgraph.checkpoint.postgres import PostgresSaver
  memory = PostgresSaver.from_conn_string(os.getenv("DATABASE_URL"))
Same interface — only the constructor changes.
"""
from langgraph.checkpoint.memory import MemorySaver
from config.settings import settings

# Shared across all sessions in this process.
# Each session is isolated by its thread_id inside the config dict.
_memory_saver = MemorySaver()


def get_memory() -> MemorySaver:
    """Return the shared MemorySaver checkpointer."""
    return _memory_saver


def get_thread_config(session_id: str) -> dict:
    """
    Build the LangGraph config dict for a session.

    Pass this as config= when calling graph.ainvoke() or graph.astream().
    LangGraph uses thread_id to load/save state for this session.
    recursion_limit caps the agent loop to prevent runaway reasoning.

    Args:
        session_id: Unique identifier for this chat session

    Returns:
        LangGraph-compatible config dict
    """
    return {
        "configurable": {"thread_id": session_id},
        # Raises GraphRecursionError if exceeded — caught in orchestrator.process_query()
        "recursion_limit": settings.max_agent_iterations,
    }
