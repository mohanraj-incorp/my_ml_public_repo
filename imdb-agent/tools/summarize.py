"""
Summarisation tool for the SemanticAgent.

When search returns multiple movies, the agent calls this to synthesise
them into a coherent narrative rather than just listing results.

Keeping summarisation in a dedicated tool (vs. letting the agent do it in its
final answer) has two benefits:
  1. The summarisation step appears in traces — visible in logs and callbacks
  2. The output guardrail can inspect the summarised text before it reaches the user

SCALE NOTE: For large batches (>20 passages), stream the summarisation response
using llm.astream() and forward chunks to the UI with st.write_stream().
"""
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

# Safety cap: don't feed more than this many chars to the summariser prompt
_MAX_PASSAGE_CHARS = 10_000


async def summarize_passages(passages: list[str], focus: str, llm) -> str:
    """
    Summarise a list of movie plot passages with a specific focus.

    Args:
        passages: List of movie description strings (from hybrid_search results)
        focus:    What angle to summarise from, e.g.:
                  "common themes and plot elements"
                  "the director's storytelling style across these films"
                  "how these movies portray grief or loss"
        llm:      LLM instance (must support ainvoke)

    Returns:
        Coherent 2–3 paragraph summary string.
    """
    if not passages:
        return "No passages provided to summarise."

    combined = "\n\n---\n\n".join(
        f"Movie {i + 1}: {p}" for i, p in enumerate(passages)
    )

    prompt = (
        f"Summarise the following movie information, focusing on: {focus}\n\n"
        f"{combined}\n\n"
        "Write a concise 2–3 paragraph summary that directly addresses the focus. "
        "Reference specific movie titles when making claims."
    )

    # Token safety: truncate if combined passages are too long
    if len(prompt) > _MAX_PASSAGE_CHARS:
        prompt = prompt[:_MAX_PASSAGE_CHARS] + "\n\n[Passages truncated]\n\nSummarise the above."
        logger.warning("Summarise prompt truncated due to length")

    response = await llm.ainvoke(prompt)
    return response.content
