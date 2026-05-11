"""
Orchestrator — the architectural core of the system.

DESIGN DECISION: The orchestrator is a LangGraph StateGraph (deterministic
routing), NOT a ReAct agent. Why this matters:

  ReAct agents decide tool calls dynamically through reasoning — good when the
  sequence of steps is unknown upfront (e.g., multi-step data exploration).

  StateGraph gives you explicit, auditable control flow — every possible path
  is declared as nodes and edges. The routing logic is a single LLM classifier
  call with a JSON response, not an open-ended reasoning loop.

  For an orchestrator whose job is routing (not exploring), StateGraph is the
  right choice: it's whiteboard-explainable, deterministic, and easier to debug.

GRAPH STRUCTURE:
  START
    │
    ▼
  classify_query  ◄─────────────────────────────────────────┐
    │                                                         │
    ├─ route="analytical" ──► run_analytical ──► recommend ──► END
    ├─ route="semantic"   ──► run_semantic   ──► recommend ──► END
    ├─ route="blocked"    ──────────────────────────────────► END
    └─ route="clarify"    ──► ask_clarification ─────────────► END
         (user replies)   ──────────────────────────────────┘
         (on next invoke, graph resumes classify_query with the clarification)

RECOMMENDATIONS NODE: After every analytical or semantic answer, a lightweight
  SQL lookup finds movies with similar IMDB rating (±0.5) and Meta score (±15)
  and appends a "You might also enjoy" block to the response. No LLM call —
  pure structured data lookup. Fires on both agent paths, skipped on clarify/blocked.

MEMORY: MemorySaver checkpointer persists AgentState per thread_id (session_id).
  Each graph.ainvoke() call loads the previous state and appends new messages.

RETRIES: tenacity retries the LLM routing call up to 3 times with exponential
  backoff — handles transient OpenAI rate limits or network errors.
"""
import json
import logging
import re
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents.analytical_agent import create_analytical_agent
from agents.semantic_agent import create_semantic_agent
from config.prompts import ORCHESTRATOR_SYSTEM
from tools.recommendation_tools import build_recommendations, _extract_primary_title
from config.settings import settings
from guardrails.input_guards import InputGuardError, run_input_guardrails
from guardrails.output_guards import run_output_guardrails
from logging_.callbacks import IMDBCallbackHandler
from logging_.schema import AgentCallLog
from memory.long_term import get_preferences
from memory.short_term import get_memory, get_thread_config

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The data that flows through every node of the graph.

    'messages' uses the add_messages reducer — new messages are APPENDED,
    not replaced, on every node update. This is what gives us conversation memory
    within a turn.

    All other fields are plain replacement (last write wins).
    """
    messages: Annotated[list[BaseMessage], add_messages]
    query: str              # Current user query (may be appended to after clarification)
    route: str              # "analytical" | "semantic" | "clarify" | "blocked"
    agent_response: str     # Final cleaned response to return to the user
    context: str            # RAG passages (for output hallucination check)
    session_id: str         # Isolates memory / logging per session
    clarification_count: int  # Tracks how many clarifying questions we've asked
    recommendations: str    # "You might also enjoy" block appended after agent_response


# ── LLM factory ────────────────────────────────────────────────────────────────

def _make_llm(session_id: str = "") -> tuple[ChatOpenAI, IMDBCallbackHandler | None]:
    """
    Create a ChatOpenAI instance with the callback handler attached.
    Returns (llm, handler) so callers can pass the handler into agent.ainvoke config,
    ensuring tool events inside LangGraph sub-agents also reach the JSONL trace.
    """
    handler = IMDBCallbackHandler(session_id) if session_id else None
    callbacks = [handler] if handler else []
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.max_tokens_per_response,
        api_key=settings.openai_api_key,
        callbacks=callbacks,
    )
    return llm, handler


# ── Routing LLM call (with retry) ──────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(settings.llm_retry_attempts),
    wait=wait_exponential(multiplier=1, max=settings.llm_retry_max_wait),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _classify_query(query: str, history: str, llm: ChatOpenAI) -> dict:
    """
    Ask the LLM to classify the query as analytical / semantic / clarify.

    Returns a dict: {"route": ..., "reason": ..., "clarification": ...}

    RETRY: tenacity retries on any exception with exponential backoff.
    Handles: rate limit errors, network timeouts, malformed JSON responses.
    After llm_retry_attempts failures the exception propagates and the node
    returns a safe error message to the user.
    """
    prompt = (
        f"{ORCHESTRATOR_SYSTEM}\n\n"
        f"Recent conversation:\n{history}\n\n"
        f"User query: {query}\n\nJSON:"
    )
    response = await llm.ainvoke(prompt)

    try:
        result = json.loads(response.content.strip())
        assert result.get("route") in ("analytical", "semantic", "clarify")
        return result
    except (json.JSONDecodeError, AssertionError):
        logger.warning(f"Malformed routing response: {response.content[:200]}")
        # Safe fallback: semantic search handles most queries reasonably well
        return {"route": "semantic", "reason": "parse-error fallback", "clarification": ""}


# ── Graph nodes ────────────────────────────────────────────────────────────────

async def classify_query_node(state: AgentState) -> dict:
    """
    Node 1: Validate input, load user preferences, classify and route the query.
    This is the only place input guardrails run.
    """
    query = state["query"]
    session_id = state.get("session_id", "unknown")
    llm, handler = _make_llm(session_id)

    # ── Input guardrails ──────────────────────────────────────────────────────
    try:
        await run_input_guardrails(query, llm)
    except InputGuardError as e:
        if handler:
            handler.log_guardrail("input", "blocked", passed=False, details=str(e)[:200])
        return {
            "route": "blocked",
            "agent_response": str(e),
            "messages": [AIMessage(content=str(e))],
        }

    # ── Inject long-term user preferences into query context ──────────────────
    prefs = await get_preferences(session_id)
    contextual_query = query
    if prefs:
        contextual_query = f"{query}\n[User preferences: {json.dumps(prefs)}]"

    # ── Build recent conversation history for context-aware routing ───────────
    recent = state.get("messages", [])[-6:]   # Last 3 turns (6 messages)
    history = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content[:200]}"
        for m in recent
        if isinstance(m, (HumanMessage, AIMessage))
    )

    if handler and recent:
        handler._write(json.dumps({
            "type": "short_term_memory",
            "session_id": session_id,
            "turns_loaded": len([m for m in recent if isinstance(m, HumanMessage)]),
            "history_preview": history[:300],
        }))

    routing = await _classify_query(contextual_query, history, llm)
    logger.info(f"[{session_id}] route={routing['route']} reason={routing.get('reason', '')}")

    if handler:
        log = AgentCallLog(
            session_id=session_id,
            agent_name="orchestrator",
            user_query=query[:200],
            routing_decision=routing["route"],
            response_summary=routing.get("reason", ""),
        )
        handler._write(log.to_json())

    updates: dict = {"route": routing["route"]}

    if routing["route"] == "clarify" and routing.get("clarification"):
        updates["messages"] = [AIMessage(content=routing["clarification"])]

    return updates


def _conversation_context(state: AgentState, n_turns: int = 3) -> list[BaseMessage]:
    """
    Return the last n_turns of HumanMessage+AIMessage pairs from state so
    sub-agents can follow references like "list them all" or "show more from 1997".
    ToolMessages are excluded — they're noisy internal plumbing, not conversation.
    """
    human_ai = [
        m for m in state.get("messages", [])
        if isinstance(m, (HumanMessage, AIMessage))
    ]
    return human_ai[-(n_turns * 2):]


async def run_analytical_node(state: AgentState) -> dict:
    """
    Node 2a: Delegate to the Analytical ReAct agent (SQL + pandas tools).
    Analytical answers don't use RAG — skip hallucination check.
    """
    session_id = state.get("session_id", "unknown")
    llm, handler = _make_llm(session_id)
    agent = create_analytical_agent(llm)
    cb_config = {"callbacks": [handler]} if handler else {}

    result = await agent.ainvoke(
        {"messages": _conversation_context(state)},
        config={"recursion_limit": settings.max_agent_iterations, **cb_config},
    )

    raw = result["messages"][-1].content if result.get("messages") else "No response generated."
    cleaned = await run_output_guardrails(raw, context="", skip_hallucination=True)

    return {
        "agent_response": cleaned,
        "messages": [AIMessage(content=cleaned)],
    }


async def run_semantic_node(state: AgentState) -> dict:
    """
    Node 2b: Delegate to the Semantic ReAct agent (hybrid search + summarise).
    Extracts retrieved context from tool messages for hallucination check.
    """
    session_id = state.get("session_id", "unknown")
    llm, handler = _make_llm(session_id)
    agent = create_semantic_agent(llm)
    cb_config = {"callbacks": [handler]} if handler else {}

    result = await agent.ainvoke(
        {"messages": _conversation_context(state)},
        config={"recursion_limit": settings.max_agent_iterations, **cb_config},
    )

    raw = result["messages"][-1].content if result.get("messages") else "No response generated."

    # Collect tool call outputs as context for hallucination check
    context = "\n\n".join(
        m.content for m in result.get("messages", []) if isinstance(m, ToolMessage)
    )[:3000]

    cleaned = await run_output_guardrails(raw, context=context, llm=llm)

    return {
        "agent_response": cleaned,
        "context": context,
        "messages": [AIMessage(content=cleaned)],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """
    Node 3: Increment the clarification counter and surface the clarification question.
    The actual clarification question was already added to messages in classify_query_node.
    Copy it into agent_response so process_query can return it to the UI.
    The graph ends after this node — LangGraph pauses and waits for the next user message.
    On next ainvoke(), the graph resumes from classify_query with the clarification context.
    """
    clarification_text = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            clarification_text = msg.content
            break

    return {
        "clarification_count": state.get("clarification_count", 0) + 1,
        "agent_response": clarification_text,
    }


def recommendations_node(state: AgentState) -> dict:
    """
    Node 4: Append "You might also enjoy" recommendations to the agent response.

    WHY A SEPARATE NODE (not inside run_analytical_node / run_semantic_node):
      Separation of concerns — each node does one job. The agent nodes answer
      the question; this node enriches the answer with similar-movie suggestions.
      It also means recommendations can be disabled by removing this node and
      its edges without touching the agent logic.

    HOW IT WORKS:
      1. Extract the primary movie title from agent_response text (regex)
      2. Look up that movie's IMDB_Rating + Meta_score in SQLite
      3. SQL query: movies within ±0.5 IMDB and ±15 Meta of the primary movie
      4. Format as a "You might also enjoy" block and append

    NO LLM CALL — pure structured data lookup. Fast, deterministic, free.

    SKIPS GRACEFULLY when:
      - Response is a list (no single primary movie to anchor on)
      - Primary movie not found in DB (title extraction failed)
      - No similar movies exist in the score range
    """
    agent_response = state.get("agent_response", "")
    if not agent_response:
        return {"recommendations": ""}

    rec_block = build_recommendations(agent_response)

    if rec_block:
        # Append recommendations to the agent response so both appear in one message
        combined = agent_response + rec_block
        return {
            "recommendations": rec_block,
            "agent_response": combined,
            "messages": [AIMessage(content=combined)],
        }

    return {"recommendations": ""}


# ── Conditional routing ────────────────────────────────────────────────────────

def route_after_classify(
    state: AgentState,
) -> Literal["analytical", "semantic", "clarify", "__end__"]:
    """
    Edge function: tells LangGraph which node to visit after classify_query_node.
    Returns node name string (must match the add_node() calls below).
    """
    route = state.get("route", "semantic")

    if route == "blocked":
        return END

    if route == "clarify":
        # Hard cap on clarification rounds to prevent looping
        if state.get("clarification_count", 0) >= 2:
            logger.warning("Max clarification rounds reached — defaulting to semantic")
            return "semantic"
        return "clarify"

    return route  # "analytical" or "semantic"


_RECOMMENDATION_INTENT_RE = re.compile(
    r"\b(recommend|suggest|similar\s+to|like\s+\w|movies?\s+like|films?\s+like"
    r"|what\s+(else|other|should\s+I\s+watch)|more\s+movies?|find\s+me\s+movies?"
    r"|based\s+on|in\s+the\s+(style|spirit|vein)\s+of)\b",
    re.IGNORECASE,
)


def _is_recommendation_query(query: str) -> bool:
    """Return True when the user is already asking for recommendations."""
    return bool(_RECOMMENDATION_INTENT_RE.search(query))


def route_after_agent(state: AgentState) -> Literal["recommendations", "__end__"]:
    """
    Conditional edge: decides whether to show recommendations after an agent answer.

    FIRES recommendations only when:
      - The response is about a single identifiable movie (title extractable), AND
      - The user was NOT already asking for recommendations (to avoid a redundant
        "You might also enjoy" block after an answer that is itself a list of suggestions)

    SKIPS for:
      - Recommendation queries ("movies like X", "suggest something similar")
      - List queries           ("top 10 movies of the 1990s")
      - Aggregations           ("how many Nolan films are in the dataset?")
      - Summarisations         ("compare Spielberg's sci-fi films")
    """
    query = state.get("query", "")
    if _is_recommendation_query(query):
        return END

    response = state.get("agent_response", "")
    if _extract_primary_title(response):
        return "recommendations"
    return END


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Assemble and compile the LangGraph StateGraph.

    Adding a new specialist agent = add_node() + add_edge() + update route_after_classify().
    The rest of the system (memory, guardrails, logging) is unchanged.

    Returns:
        Compiled LangGraph app (call with graph.ainvoke(state, config=config))
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify_query", classify_query_node)
    graph.add_node("analytical", run_analytical_node)
    graph.add_node("semantic", run_semantic_node)
    graph.add_node("clarify", ask_clarification_node)
    graph.add_node("recommendations", recommendations_node)

    graph.add_edge(START, "classify_query")

    graph.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "analytical": "analytical",
            "semantic": "semantic",
            "clarify": "clarify",
            END: END,
        },
    )

    # Conditional: only route to recommendations when the response is about
    # a single identifiable movie. List/aggregation/summary answers go to END directly.
    graph.add_conditional_edges(
        "analytical",
        route_after_agent,
        {"recommendations": "recommendations", END: END},
    )
    graph.add_conditional_edges(
        "semantic",
        route_after_agent,
        {"recommendations": "recommendations", END: END},
    )
    graph.add_edge("recommendations", END)

    # Clarification ends the turn immediately — no recommendations for a question
    graph.add_edge("clarify", END)

    return graph.compile(checkpointer=get_memory())


# Module-level compiled graph — built once, shared across all requests
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public entry point ─────────────────────────────────────────────────────────

async def process_query(query: str, session_id: str) -> str:
    """
    Process one user message through the full orchestration graph.

    Args:
        query:      Raw user message text
        session_id: Unique session ID (scopes memory, logs, and preferences)

    Returns:
        Agent response string (cleaned by output guardrails)
    """
    graph = get_graph()
    config = get_thread_config(session_id)

    initial_state = {
        "query": query,
        "session_id": session_id,
        "messages": [HumanMessage(content=query)],
        "route": "",
        "agent_response": "",
        "context": "",
        "clarification_count": 0,
        "recommendations": "",
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)
        return result.get("agent_response") or "I couldn't generate a response. Please try again."

    except GraphRecursionError:  # noqa: E722
        logger.error(f"[{session_id}] GraphRecursionError — recursion limit exceeded")
        return (
            "I got stuck in a reasoning loop. Please try rephrasing your question "
            "or breaking it into smaller parts."
        )

    except Exception as e:
        logger.error(f"[{session_id}] Unexpected error: {e}", exc_info=True)
        return f"An unexpected error occurred. Please try again. (Details: {str(e)[:100]})"


async def process_query_full(query: str, session_id: str) -> dict:
    """
    Evaluation variant of process_query — returns answer AND retrieved context.

    Used by the evaluation pipeline so RAGAS receives the actual document chunks
    retrieved by the semantic agent, not the placeholder keywords in the golden dataset.
    Not used by the Streamlit app (which only needs the answer string).

    Returns:
        {"answer": str, "context": list[str], "route": str}
    """
    graph = get_graph()
    config = get_thread_config(session_id)

    initial_state = {
        "query": query,
        "session_id": session_id,
        "messages": [HumanMessage(content=query)],
        "route": "",
        "agent_response": "",
        "context": "",
        "clarification_count": 0,
        "recommendations": "",
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)
        answer = result.get("agent_response") or "I couldn't generate a response. Please try again."
        raw_context = result.get("context", "")
        # Split the joined context back into individual chunks for RAGAS
        context_chunks = [c.strip() for c in raw_context.split("\n\n") if c.strip()] if raw_context else []
        return {
            "answer": answer,
            "context": context_chunks,
            "route": result.get("route", "semantic"),
        }

    except Exception as e:
        logger.error(f"[{session_id}] process_query_full error: {e}", exc_info=True)
        return {"answer": f"ERROR: {e}", "context": [], "route": "error"}
