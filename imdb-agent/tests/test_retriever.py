"""
Integration tests for the hybrid retrieval pipeline.

These tests require the indexes to be built first:
    python scripts/build_indexes.py

Skip gracefully if indexes don't exist (CI without data).
"""
import os
import pytest
from config.settings import settings

# Skip entire module if indexes not built
pytestmark = pytest.mark.skipif(
    not os.path.exists(settings.bm25_index_path),
    reason="Indexes not built — run: python scripts/build_indexes.py",
)


@pytest.mark.asyncio
async def test_hybrid_search_returns_results():
    from rag.retriever import HybridRetriever
    retriever = HybridRetriever()
    retriever.initialize()

    results = await retriever.retrieve("psychological thriller", top_k=5)
    assert len(results) > 0
    assert all("title" in doc for doc in results)


@pytest.mark.asyncio
async def test_director_filter_applies():
    from rag.retriever import HybridRetriever
    retriever = HybridRetriever()
    retriever.initialize()

    results = await retriever.retrieve(
        "science fiction", filter_director="Spielberg", top_k=5
    )
    # All results should be Spielberg films
    for doc in results:
        assert "spielberg" in doc.get("director", "").lower()


@pytest.mark.asyncio
async def test_empty_results_for_nonsense_query():
    from rag.retriever import HybridRetriever
    retriever = HybridRetriever()
    retriever.initialize()

    # With director filter for nonexistent director, results should be empty
    results = await retriever.retrieve(
        "movie", filter_director="XXXXXXXXXNONEXISTENT", top_k=5
    )
    assert len(results) == 0


def test_rrf_fusion_score_ordering():
    """RRF should rank documents appearing in both lists higher."""
    from rag.retriever import HybridRetriever
    retriever = HybridRetriever()

    doc_a = {"id": 1, "text": "a", "title": "A"}
    doc_b = {"id": 2, "text": "b", "title": "B"}
    doc_c = {"id": 3, "text": "c", "title": "C"}

    # doc_a appears in both lists at rank 1 — should score highest
    bm25_results = [(doc_a, 1.0), (doc_b, 0.5)]
    vector_results = [(doc_a, 0.1), (doc_c, 0.2)]

    fused = retriever._rrf_fuse(bm25_results, vector_results)
    assert fused[0]["id"] == 1, "doc_a should be ranked first (appears in both lists)"
