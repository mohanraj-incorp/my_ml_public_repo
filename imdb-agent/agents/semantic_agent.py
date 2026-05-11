"""
Semantic Agent — answers theme, plot, and similarity queries using RAG.

Uses create_react_agent with two tools:
  hybrid_search     — BM25 + vector + reranking (the RAG retrieval step)
  summarize_results — synthesise multiple movie descriptions into a narrative

WHY ASYNC TOOLS: hybrid_search runs BM25 + FAISS concurrently via asyncio.gather().
  LangGraph's create_react_agent supports async tools natively — the agent just
  awaits tool calls. No threading tricks needed.

DESIGN NOTE: The semantic agent never touches SQLite directly. If a semantic
  query also needs exact aggregations (e.g., "Nolan's sci-fi movies sorted by rating"),
  the orchestrator routes it here AND the agent has access only to search + summarise.
  For hybrid queries needing both, the orchestrator could run both agents and merge —
  but for this scope we route to whichever agent is primary.
"""
import logging
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config.prompts import SEMANTIC_SYSTEM
from tools.search_tools import hybrid_search as _hybrid_search
from tools.summarize import summarize_passages as _summarize

logger = logging.getLogger(__name__)


# ── Tool definitions ────────────────────────────────────────────────────────────

@tool
async def semantic_search(
    query: str,
    filter_director: str = "",
    filter_genre: str = "",
    top_k: int = 10,
    sort_by: str = "relevance",
) -> str:
    """
    Search IMDB movies by semantic meaning using hybrid BM25 + vector search.

    Finds movies based on plot themes, mood, or description — not just exact keywords.
    Each result includes the IMDB rating and Meta score so you can answer hybrid
    queries (e.g. "vengeance movies ranked by IMDB rating") in a single call.

    Args:
        query:           Natural-language description of what you're looking for.
                         Examples: "dystopian sci-fi with AI themes",
                                   "heist movies where the plan goes wrong",
                                   "films about grief and loss"
        filter_director: Optional director name to narrow results
                         (e.g. "Nolan", "Spielberg")
        filter_genre:    Optional genre filter (e.g. "Drama", "Action", "Comedy")
        top_k:           Number of results to return (max 10)
        sort_by:         Order results by:
                           "relevance"   — most thematically relevant first (default)
                           "imdb_rating" — highest IMDB rating first
                           "meta_score"  — highest Metacritic score first
                           "year"        — most recently released first
                         Use "imdb_rating" or "year" for queries like
                         "vengeance movies ranked by rating".
    """
    return await _hybrid_search(query, filter_director, filter_genre, top_k, sort_by)


@tool
async def summarize_results(passages: list[str], focus: str) -> str:
    """
    Summarise a list of movie descriptions into a coherent narrative.

    Call this after semantic_search when the user wants a synthesis, not a list.
    Example: "Summarise what makes these movies similar" or
             "Describe the director's recurring themes across these films"

    Args:
        passages: List of movie plot/description strings from search results
        focus:    What aspect to highlight in the summary
    """
    # The LLM is injected at agent-creation time via the closure below
    # This pattern avoids passing the LLM as a tool argument (which would expose it)
    raise NotImplementedError("Use _make_summarize_tool() to inject the LLM")


def create_semantic_agent(llm: ChatOpenAI):
    """
    Build and return the semantic ReAct agent.

    The summarize tool needs access to the LLM (for generating the summary).
    We inject it via a closure rather than making it a tool parameter.

    Args:
        llm: Configured ChatOpenAI instance (with callbacks attached)

    Returns:
        Compiled LangGraph ReAct agent
    """

    @tool
    async def summarize_movie_results(passages: list[str], focus: str) -> str:
        """
        Summarise retrieved movie plot descriptions into a focused narrative.

        Call after semantic_search to synthesise results rather than just listing them.
        Useful for: "What themes do these movies share?", "Describe Nolan's style"

        Args:
            passages: List of movie description strings (from search results)
            focus:    What to highlight: e.g., "common themes", "visual storytelling style"
        """
        return await _summarize(passages, focus, llm)

    tools = [semantic_search, summarize_movie_results]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SEMANTIC_SYSTEM,
    )

    logger.info("Semantic agent created with tools: semantic_search, summarize_movie_results")
    return agent
