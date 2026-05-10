"""
Policy FAQ tool — delegates to the hybrid RAG retriever.
Returns ranked document chunks that the Policy FAQ agent uses to answer
the prospect's question. Agent is responsible for synthesizing the answer.
"""
from rag.retriever import hybrid_retrieve


async def search_policy_docs(query: str, property_id: str | None = None) -> list[dict]:
    """
    Retrieves the most relevant policy chunks for a question.
    Scoped to property_id if provided; falls back to global docs if
    property-specific results are insufficient.
    """
    results = await hybrid_retrieve(query=query, property_id=property_id, top_k=5)
    return results
