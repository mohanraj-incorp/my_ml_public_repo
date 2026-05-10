"""
Hybrid RAG retriever: combines dense (Vertex AI Vector Search) and sparse (BM25)
retrieval then re-ranks the merged results with a cross-encoder.

Why hybrid? Dense retrieval handles paraphrased questions well ("leave early" →
finds early termination clause). BM25 handles exact legal terms well ("Section 4.2").
Combining both covers more ground than either alone.
"""
from google.cloud import aiplatform
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config.settings import settings

# Cross-encoder for re-ranking — loaded once at module import time
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


async def dense_retrieve(query: str, property_id: str | None, top_k: int) -> list[dict]:
    """
    Queries Vertex AI Vector Search with optional property_id filter.
    Returns chunks with their text content and metadata.
    """
    aiplatform.init(project=settings.gcp_project_id, location=settings.gcp_region)

    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=settings.vector_search_index_endpoint
    )

    # Build metadata filter to scope results to this property + global docs
    # "scope:global" ensures Fair Housing Act and lease templates always appear
    allowed_property_ids = ["GLOBAL"]
    if property_id:
        allowed_property_ids.append(property_id)

    restricts = [
        {"namespace": "property_id", "allow_tokens": allowed_property_ids}
    ]

    # Embed the query using Vertex AI text embeddings
    from vertexai.language_models import TextEmbeddingModel
    embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    query_embedding = embed_model.get_embeddings([query])[0].values

    response = index_endpoint.find_neighbors(
        deployed_index_id=settings.vector_search_deployed_index_id,
        queries=[query_embedding],
        num_neighbors=top_k,
        restricts=restricts,
    )

    results = []
    for neighbor in response[0]:
        results.append({
            "chunk_id": neighbor.id,
            "score": neighbor.distance,
            "text": neighbor.restricts.get("text", ""),   # text stored as restrict metadata
            "property_id": neighbor.restricts.get("property_id", "GLOBAL"),
            "policy_type": neighbor.restricts.get("policy_type", ""),
            "source_file": neighbor.restricts.get("source_file", ""),
        })

    return results


def sparse_retrieve(query: str, corpus: list[dict], top_k: int) -> list[dict]:
    """
    BM25 retrieval over the same corpus returned by dense search.
    In production the BM25 index would be pre-built over all chunks.
    For the POC we run BM25 over the dense results to keep it simple.
    """
    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair each doc with its BM25 score and sort
    scored = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Cross-encoder re-ranks the merged candidate list.
    Produces a single ordered list where the most relevant chunk is first.
    """
    pairs = [(query, doc["text"]) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


async def hybrid_retrieve(query: str, property_id: str | None, top_k: int = 5) -> list[dict]:
    """
    Full hybrid pipeline:
    1. Dense retrieval from Vertex AI Vector Search (property-scoped)
    2. Sparse BM25 retrieval over the dense result set
    3. Merge unique results from both
    4. Cross-encoder re-ranking
    5. Return top_k most relevant chunks
    """
    dense_results = await dense_retrieve(query, property_id, top_k=top_k * 2)

    if not dense_results:
        return []

    sparse_results = sparse_retrieve(query, dense_results, top_k=top_k * 2)

    # Merge: union of dense + sparse, deduplicated by chunk_id
    seen = set()
    merged = []
    for doc in dense_results + sparse_results:
        if doc["chunk_id"] not in seen:
            seen.add(doc["chunk_id"])
            merged.append(doc)

    return rerank(query, merged, top_k=top_k)
