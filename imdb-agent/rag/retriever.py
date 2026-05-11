"""
Hybrid retrieval pipeline: BM25 sparse + FAISS dense, fused with RRF, then reranked.

PIPELINE OVERVIEW:
  Query
    ├─ BM25 (top 50)  ──┐
    └─ FAISS (top 50) ──┴─► RRF fusion (up to 100 unique docs)
                               │
                          filter by director/genre (optional, post-retrieval)
                               │
                        Cross-encoder rerank (top 20 → top 10)
                               │
                           Final results

WHY HYBRID over BM25-only or vector-only:
  BM25 excels at keyword matching: exact names, years, uncommon terms.
  Vector search excels at semantic similarity: "movies about grief" even when
  the word "grief" never appears in plot text.
  Together they catch what either misses alone.

WHY RRF (Reciprocal Rank Fusion):
  Simple formula, no hyperparameters to tune beyond k=60 (from the original paper).
  Consistently outperforms linear score combination in retrieval benchmarks.
  Formula: score(d) = Σ_i  1 / (k + rank_i(d))

ASYNC: BM25 and FAISS retrieval run concurrently with asyncio.gather().
In a single-user demo this saves ~10–50ms; in a multi-user prod system this
prevents one slow retrieval from blocking other requests.
"""
import asyncio
import logging
import os
import pickle

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.settings import settings
from rag.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Stateful retriever — load indexes once at startup, reuse across all queries.
    Initialised lazily on first retrieve() call (or explicitly via initialize()).
    """

    def __init__(self):
        self._initialized = False
        self.bm25: BM25Okapi = None
        self.bm25_docs: list[dict] = None
        self.faiss_index: faiss.Index = None
        self.faiss_docs: list[dict] = None
        self.embed_model: SentenceTransformer = None
        self.reranker: CrossEncoderReranker = None

    def initialize(self) -> None:
        """Load indexes from disk. Safe to call multiple times (idempotent)."""
        if self._initialized:
            return

        logger.info("Loading BM25 index…")
        with open(settings.bm25_index_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_docs = bm25_data["docs"]

        logger.info("Loading FAISS index…")
        self.faiss_index = faiss.read_index(
            os.path.join(settings.faiss_index_path, "index.faiss")
        )
        with open(os.path.join(settings.faiss_index_path, "docs.pkl"), "rb") as f:
            self.faiss_docs = pickle.load(f)

        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.embed_model = SentenceTransformer(settings.embedding_model)

        self.reranker = CrossEncoderReranker()
        self._initialized = True
        logger.info("HybridRetriever ready.")

    # ── Sparse retrieval (BM25) ────────────────────────────────────────────────

    def _bm25_retrieve(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        """Return (doc, bm25_score) pairs sorted by score descending."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.bm25_docs[i], float(scores[i])) for i in top_indices]

    # ── Dense retrieval (FAISS) ────────────────────────────────────────────────

    def _vector_retrieve(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        """Return (doc, l2_distance) pairs. Lower L2 = more similar."""
        vec = self.embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.faiss_index.search(vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # FAISS returns -1 when fewer results exist than top_k
                results.append((self.faiss_docs[idx], float(dist)))
        return results

    # ── RRF fusion ─────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        bm25_results: list[tuple[dict, float]],
        vector_results: list[tuple[dict, float]],
    ) -> list[dict]:
        """
        Merge two ranked lists into one using Reciprocal Rank Fusion.
        score(d) = Σ_i  1 / (k + rank_i(d))    k = settings.rrf_k (default 60)
        Documents absent from a list contribute 0 from that retriever.
        """
        k = settings.rrf_k
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_id = str(doc["id"])
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

        for rank, (doc, _) in enumerate(vector_results, start=1):
            doc_id = str(doc["id"])
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[did] for did in sorted_ids]

    # ── Public API ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_director: str = None,
        filter_genre: str = None,
    ) -> list[dict]:
        """
        Run hybrid retrieval asynchronously: BM25 + FAISS in parallel → RRF → rerank.

        Args:
            query:           Natural-language search query
            top_k:           Final results to return after reranking
            filter_director: Optional post-retrieval director filter
            filter_genre:    Optional post-retrieval genre filter

        Returns:
            Ranked list of doc dicts (most relevant first)

        ASYNC NOTE: asyncio.to_thread wraps CPU-bound calls so they don't block
        the event loop. Both retrievals run concurrently via asyncio.gather().
        """
        if not self._initialized:
            self.initialize()

        top_k = top_k or settings.rerank_top_n

        # Run BM25 and vector retrieval concurrently
        bm25_task = asyncio.to_thread(self._bm25_retrieve, query, settings.bm25_top_k)
        vec_task = asyncio.to_thread(self._vector_retrieve, query, settings.vector_top_k)
        bm25_results, vector_results = await asyncio.gather(bm25_task, vec_task)

        logger.debug(f"BM25={len(bm25_results)}, vector={len(vector_results)}")

        # Fuse with RRF
        fused = self._rrf_fuse(bm25_results, vector_results)

        # Post-retrieval metadata filters (applied before reranking to save reranker time)
        if filter_director:
            fused = [d for d in fused if filter_director.lower() in d.get("director", "").lower()]
        if filter_genre:
            fused = [d for d in fused if filter_genre.lower() in d.get("genre", "").lower()]

        # Rerank top-20 candidates with cross-encoder → top_k final results
        candidates = fused[:20]
        reranked = await asyncio.to_thread(self.reranker.rerank, query, candidates, top_k)

        logger.info(
            f"Hybrid retrieve: fused={len(fused)} → reranked={len(reranked)} "
            f"| query='{query[:50]}'"
        )
        return reranked


# Module-level singleton — initialise once at startup, share across all requests
retriever = HybridRetriever()
