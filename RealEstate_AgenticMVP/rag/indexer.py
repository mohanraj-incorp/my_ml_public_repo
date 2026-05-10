"""
Policy document indexer — run this once to ingest PDFs into Vertex AI Vector Search.

What it does:
1. Downloads policy PDFs from GCS
2. Splits them into chunks (500 tokens, 50-token overlap)
3. Embeds each chunk using Vertex AI text-embedding-004
4. Upserts chunks into Vertex AI Vector Search with metadata (property_id, policy_type, scope)

Usage: python -m rag.indexer
"""
import asyncio
import json
import uuid
from pathlib import Path
from google.cloud import storage, aiplatform
from vertexai.language_models import TextEmbeddingModel
from config.settings import settings


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def build_metadata(property_id: str | None, policy_type: str, scope: str, source_file: str) -> dict:
    return {
        "property_id": property_id or "GLOBAL",
        "policy_type": policy_type,
        "scope": scope,
        "source_file": source_file,
    }


async def index_document(
    text: str,
    property_id: str | None,
    policy_type: str,
    scope: str,
    source_file: str,
):
    """Chunks, embeds, and upserts one document into Vertex AI Vector Search."""
    aiplatform.init(project=settings.gcp_project_id, location=settings.gcp_region)
    embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    chunks = chunk_text(text)
    datapoints = []

    for i, chunk in enumerate(chunks):
        embedding = embed_model.get_embeddings([chunk])[0].values
        metadata = build_metadata(property_id, policy_type, scope, source_file)

        datapoints.append({
            "id": f"{source_file}_{i}",
            "embedding": embedding,
            # Metadata is stored as restricts so it's filterable at query time
            "restricts": [
                {"namespace": k, "allow_tokens": [v]}
                for k, v in metadata.items()
            ] + [{"namespace": "text", "allow_tokens": [chunk[:200]]}],  # store excerpt
        })

    index = aiplatform.MatchingEngineIndex(
        index_name=settings.vector_search_index_endpoint.split("/indexEndpoints/")[0]
    )
    index.upsert_datapoints(datapoints=datapoints)
    print(f"Indexed {len(datapoints)} chunks from {source_file}")


async def run_indexer():
    """
    Entry point — indexes all sample policy docs from GCS.
    Expects GCS layout:
      gs://policy-docs-bucket/global/*.txt
      gs://policy-docs-bucket/properties/PROP_001/*.txt
    """
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(settings.gcs_policy_docs_bucket)

    for blob in bucket.list_blobs():
        text = blob.download_as_text()
        parts = Path(blob.name).parts

        if parts[0] == "global":
            await index_document(
                text=text,
                property_id=None,
                policy_type=Path(blob.name).stem,
                scope="global",
                source_file=blob.name,
            )
        elif parts[0] == "properties" and len(parts) >= 3:
            property_id = parts[1]
            await index_document(
                text=text,
                property_id=property_id,
                policy_type=Path(blob.name).stem,
                scope="property",
                source_file=blob.name,
            )


if __name__ == "__main__":
    asyncio.run(run_indexer())
