"""
Build and persist FAISS vector index + BM25 index from the IMDB CSV.

Run once before starting the app (or via scripts/build_indexes.py):
    python scripts/build_indexes.py

WHAT WE INDEX: The 'Overview' (plot) field is the primary semantic signal,
combined with Series_Title and Genre so hybrid search can match on titles and
genre keywords as well as plot themes.

SCALE NOTE: For datasets 10x larger, stream the CSV in chunks:
    for chunk in pd.read_csv(path, chunksize=1000): ...
and call faiss_index.add(embeddings) incrementally. For real-time updates
(new movies), call add() then faiss.write_index() — no full rebuild needed.
"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)


def _build_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convert dataframe rows to searchable document dicts.
    Combines title + genre + overview into one text field for indexing.
    Field order matters for BM25 term frequency weighting.

    IMPORTANT — index alignment:
    df must be reset_index(drop=True) before calling this so that
    doc["id"] == SQLite movie_id for every row. Both indexer.py and
    sqlite_tools.py apply the same dropna+reset_index to guarantee this.
    """
    docs = []
    for idx, row in df.iterrows():
        text = (
            f"{row.get('Series_Title', '')} "
            f"{row.get('Genre', '')} "
            f"{row.get('Overview', '')}"
        ).strip()
        docs.append({
            "id": int(idx),                          # shared key → SQLite movie_id
            "title": str(row.get("Series_Title", "")),
            "year": str(row.get("Released_Year", "")),
            "genre": str(row.get("Genre", "")),
            "director": str(row.get("Director", "")),
            "overview": str(row.get("Overview", "")),
            # Structured metrics stored as metadata so the semantic agent can
            # sort/filter RAG results without a separate SQLite round-trip.
            "imdb_rating": float(row["IMDB_Rating"]) if pd.notna(row.get("IMDB_Rating")) else None,
            "meta_score": float(row["Meta_score"]) if pd.notna(row.get("Meta_score")) else None,
            "text": text,
        })
    return docs


def build_bm25_index(docs: list[dict]) -> BM25Okapi:
    """BM25Okapi index from tokenised document texts. Simple word-split is fine for movie plots."""
    tokenised = [doc["text"].lower().split() for doc in docs]
    return BM25Okapi(tokenised)


def build_faiss_index(
    docs: list[dict], model: SentenceTransformer
) -> tuple[faiss.Index, np.ndarray]:
    """
    Encode docs with sentence-transformers and build a FAISS flat L2 index.

    IndexFlatL2 = exact nearest-neighbour search (no approximation).
    Correct choice for IMDB's 1000 docs — exact search is fast enough.

    SCALE NOTE: For 100k+ docs, use IndexIVFFlat (inverted file) or IndexHNSWFlat
    for sub-linear search time at a small accuracy cost.
    """
    texts = [doc["text"] for doc in docs]
    logger.info(f"Encoding {len(texts)} documents with {settings.embedding_model}…")
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, embeddings


def build_and_save_indexes(
    csv_path: str = None, force_rebuild: bool = False
) -> None:
    """
    Load CSV → build BM25 + FAISS indexes → save to disk.

    Args:
        csv_path: Override path for IMDB CSV.
        force_rebuild: If True, rebuild even if index files already exist.
                       Useful after updating the dataset.
    """
    csv_path = csv_path or settings.csv_path
    bm25_path = settings.bm25_index_path
    faiss_path = settings.faiss_index_path

    if (
        not force_rebuild
        and os.path.exists(bm25_path)
        and os.path.exists(os.path.join(faiss_path, "index.faiss"))
    ):
        logger.info("Indexes already exist — skipping rebuild. Pass force_rebuild=True to override.")
        return

    logger.info(f"Loading dataset from {csv_path}…")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Overview"]).reset_index(drop=True)
    docs = _build_documents(df)
    logger.info(f"Built {len(docs)} documents from {len(df)} movies")

    # ── BM25 ──────────────────────────────────────────────────────────────────
    logger.info("Building BM25 index…")
    bm25 = build_bm25_index(docs)
    with open(bm25_path, "wb") as f:
        # Save docs alongside index so retriever can map scores back to metadata
        pickle.dump({"bm25": bm25, "docs": docs}, f)
    logger.info(f"BM25 index saved → {bm25_path}")

    # ── FAISS ─────────────────────────────────────────────────────────────────
    logger.info("Building FAISS index…")
    model = SentenceTransformer(settings.embedding_model)
    faiss_index, _ = build_faiss_index(docs, model)

    os.makedirs(faiss_path, exist_ok=True)
    faiss.write_index(faiss_index, os.path.join(faiss_path, "index.faiss"))
    with open(os.path.join(faiss_path, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)
    logger.info(f"FAISS index saved → {faiss_path}/")
