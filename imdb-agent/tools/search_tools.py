"""
Semantic search tool — thin wrapper around HybridRetriever for the SemanticAgent.

Now includes:
  - imdb_rating and meta_score in every result (stored in doc metadata at index time)
  - sort_by parameter so the agent can answer hybrid queries like
    "find vengeance movies ranked by IMDB rating" in a single tool call

This solves the cross-agent coordination problem: the semantic agent no longer
needs to call SQLite to get ratings for its retrieved docs — the data is already
in the RAG metadata.
"""
import logging
from rag.retriever import retriever

logger = logging.getLogger(__name__)

# Valid sort options — keep this small so the LLM uses them correctly
_SORT_OPTIONS = {"relevance", "imdb_rating", "meta_score", "year"}


async def hybrid_search(
    query: str,
    filter_director: str = "",
    filter_genre: str = "",
    top_k: int = 10,
    sort_by: str = "relevance",
) -> str:
    """
    Search IMDB movies by semantic meaning using BM25 + vector + reranking.

    Finds movies based on plot themes and descriptions — not just exact keywords.
    Results include IMDB rating and Meta score so the agent can answer hybrid
    queries like "vengeance movies ranked by IMDB rating" in one call.

    Args:
        query:           Natural-language description (e.g. "movies about revenge")
        filter_director: Optional director name filter applied after retrieval
        filter_genre:    Optional genre filter (e.g. "Drama", "Action")
        top_k:           Number of results to return (max 10 recommended)
        sort_by:         How to order results after retrieval:
                           "relevance"   — cross-encoder relevance score (default)
                           "imdb_rating" — highest IMDB rating first
                           "meta_score"  — highest Metacritic score first
                           "year"        — most recent release first

    Returns:
        Formatted string table of matching movies. Each entry includes title,
        year, IMDB rating, Meta score, genre, director, and plot summary.
    """
    if not retriever._initialized:
        try:
            retriever.initialize()
        except Exception as e:
            return (
                f"Search index not available: {e}. "
                "Run 'python scripts/build_indexes.py' first."
            )

    docs = await retriever.retrieve(
        query=query,
        top_k=top_k,
        filter_director=filter_director or None,
        filter_genre=filter_genre or None,
    )

    if not docs:
        return "No movies found matching your search criteria. Try a broader description."

    # Sort by a structured field if requested.
    # The cross-encoder already sorted by relevance — this re-sorts that shortlist
    # by a numeric attribute, giving "thematically relevant AND highly rated" results.
    if sort_by in _SORT_OPTIONS and sort_by != "relevance":
        reverse = True  # all current sort options are "highest first"
        if sort_by == "imdb_rating":
            docs = sorted(docs, key=lambda d: float(d.get("imdb_rating") or 0), reverse=reverse)
        elif sort_by == "meta_score":
            docs = sorted(docs, key=lambda d: float(d.get("meta_score") or 0), reverse=reverse)
        elif sort_by == "year":
            docs = sorted(docs, key=lambda d: int(d.get("year") or 0), reverse=reverse)

    sort_label = {
        "relevance": "relevance",
        "imdb_rating": "IMDB rating ↓",
        "meta_score": "Meta score ↓",
        "year": "year (newest first)",
    }.get(sort_by, "relevance")

    header = f"Results sorted by {sort_label}:\n"
    formatted = []
    for i, doc in enumerate(docs, 1):
        rating = doc.get("imdb_rating")
        meta = doc.get("meta_score")
        rating_str = f"IMDB {rating:.1f}" if rating is not None else "IMDB N/A"
        meta_str = f"Meta {int(meta)}" if meta is not None else "Meta N/A"

        formatted.append(
            f"{i}. **{doc['title']}** ({doc.get('year', 'N/A')})  "
            f"{rating_str} | {meta_str}\n"
            f"   Genre: {doc.get('genre', 'N/A')} | Director: {doc.get('director', 'N/A')}\n"
            f"   Plot: {doc.get('overview', 'No plot available.')}"
        )

    return header + "\n\n".join(formatted)
