"""
Cross-encoder reranker for the second stage of the retrieval pipeline.

WHY TWO STAGES (bi-encoder retrieval → cross-encoder reranking):
- Bi-encoders (used in FAISS retrieval) encode query and document separately for speed.
  They run once per document at index time and can retrieve thousands of candidates fast.
- Cross-encoders encode (query, document) jointly — much more accurate because they
  model token-level interactions between query and passage — but too slow to run on
  the entire corpus.
- Solution: wide retrieval (top-50 each from BM25 + vector) → accurate reranking
  on the shortlist (top-20 candidates → top-10 final results).

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Free, runs on CPU, trained on MS MARCO passage relevance data
  - No extra API key (unlike Cohere Rerank)
  - Fast enough for 20-doc shortlists on CPU (~50–100ms)

SCALE NOTE: For latency-sensitive prod, run on GPU (10x faster) or swap for
Cohere Rerank API (one HTTP call, ~50ms, slightly better accuracy).
"""
import logging
from sentence_transformers import CrossEncoder
from config.settings import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Wraps a cross-encoder model to rerank a shortlist of candidate documents."""

    def __init__(self):
        logger.info(f"Loading cross-encoder: {settings.reranker_model}")
        self.model = CrossEncoder(settings.reranker_model)

    def rerank(self, query: str, docs: list[dict], top_n: int) -> list[dict]:
        """
        Score each (query, doc_text) pair and return the top_n by relevance.

        Args:
            query: The user's natural-language search query
            docs:  Candidate documents — each must have a 'text' field
            top_n: How many documents to return

        Returns:
            Reranked subset of docs, length ≤ top_n, highest relevance first
        """
        if not docs:
            return []

        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.model.predict(pairs)  # returns np.ndarray of floats

        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top = [doc for doc, _ in scored[:top_n]]

        logger.debug(
            f"Reranker: {len(docs)} candidates → {len(top)} results "
            f"(top score={float(max(scores)):.3f})"
        )
        return top
