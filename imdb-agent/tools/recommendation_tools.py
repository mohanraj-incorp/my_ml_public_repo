"""
Movie recommendation tool based on rating similarity.

Logic:
  1. Extract the primary movie title from the agent's response text
  2. Look up that movie's IMDB_Rating and Meta_score in SQLite
  3. Query for movies with similar scores (within configurable tolerances)
  4. Return formatted recommendations, excluding the primary movie itself

WHY RATING-BASED SIMILARITY (vs. embedding similarity):
  The requirement is specifically "similar Meta scores and IMDB ratings" —
  this is a structured data problem, not a semantic one. SQL range queries
  are the right tool: fast, exact, and fully explainable.

  Embedding-based similarity (same director, similar plot) would go through
  the HybridRetriever — a different, complementary approach.

SCALE NOTE: For a real recommendation engine at scale, replace this SQL
range query with collaborative filtering (e.g., ALS matrix factorisation)
or a graph-based approach (users who liked X also liked Y). For IMDB data
without user interaction history, rating/meta similarity is appropriate.
"""
import re
import sqlite3
import logging

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# Tolerances for "similar" — tunable via settings in production
RATING_TOLERANCE = 0.5    # ±0.5 IMDB rating points
META_TOLERANCE = 15       # ±15 Meta score points
MAX_RECOMMENDATIONS = 5


def _extract_primary_title(response_text: str) -> str | None:
    """
    Extract the most likely primary movie title from the agent's response.

    Strategy: look for the first bolded title in Markdown format (**Title**),
    which is how both agents format their primary result. Falls back to a
    quoted title pattern if no bold is found.

    Returns None if no title can be extracted — caller should skip recommendations.
    """
    # Pattern 1: **Title** (Year) — our agent output format
    bold_match = re.search(r"\*\*([^*]+)\*\*\s*\(\d{4}\)", response_text)
    if bold_match:
        return bold_match.group(1).strip()

    # Pattern 2: "Title" (Year) — common in analytical responses
    quoted_match = re.search(r'"([^"]+)"\s*\(\d{4}\)', response_text)
    if quoted_match:
        return quoted_match.group(1).strip()

    # Pattern 3: Title (Year) — unformatted, grab the first occurrence
    plain_match = re.search(r"([A-Z][^.!?\n]{3,50})\s*\((\d{4})\)", response_text)
    if plain_match:
        return plain_match.group(1).strip()

    return None


def _lookup_movie_scores(title: str) -> dict | None:
    """
    Look up IMDB_Rating and Meta_score for a movie by title (fuzzy match).
    Returns None if movie not found or scores are missing.
    """
    try:
        conn = sqlite3.connect(settings.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT Series_Title, IMDB_Rating, Meta_score, Genre
            FROM movies
            WHERE LOWER(Series_Title) LIKE LOWER(?)
            LIMIT 1
            """,
            (f"%{title}%",),
        )
        row = cursor.fetchone()
        conn.close()

        if row and row[1] is not None and row[2] is not None:
            return {
                "title": row[0],
                "imdb_rating": float(row[1]),
                "meta_score": float(row[2]),
                "genre": row[3],
            }
    except Exception as e:
        logger.warning(f"Movie lookup failed for '{title}': {e}")

    return None


def get_similar_movies(
    imdb_rating: float,
    meta_score: float,
    exclude_title: str,
    limit: int = MAX_RECOMMENDATIONS,
) -> list[dict]:
    """
    Find movies with similar IMDB rating and Meta score.

    Args:
        imdb_rating:   Reference IMDB rating (centre of range)
        meta_score:    Reference Meta score (centre of range)
        exclude_title: Title to exclude from results (the primary movie)
        limit:         Max results to return

    Returns:
        List of movie dicts sorted by IMDB rating descending
    """
    try:
        conn = sqlite3.connect(settings.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT Series_Title, Released_Year, IMDB_Rating, Meta_score, Genre, Director
            FROM movies
            WHERE IMDB_Rating BETWEEN ? AND ?
              AND Meta_score BETWEEN ? AND ?
              AND LOWER(Series_Title) != LOWER(?)
              AND IMDB_Rating IS NOT NULL
              AND Meta_score IS NOT NULL
            ORDER BY IMDB_Rating DESC
            LIMIT ?
            """,
            (
                imdb_rating - RATING_TOLERANCE,
                imdb_rating + RATING_TOLERANCE,
                meta_score - META_TOLERANCE,
                meta_score + META_TOLERANCE,
                exclude_title,
                limit,
            ),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "title": r[0],
                "year": r[1],
                "imdb_rating": r[2],
                "meta_score": r[3],
                "genre": r[4],
                "director": r[5],
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Similarity query failed: {e}")
        return []


def build_recommendations(response_text: str) -> str:
    """
    Main entry point: extract primary movie → look up scores → find similar movies.

    Args:
        response_text: The agent's answer text

    Returns:
        Formatted recommendation block to append to the response,
        or empty string if no recommendations can be generated
        (e.g., response is a list query with no single primary movie).
    """
    # Step 1: find the primary movie the response is about
    title = _extract_primary_title(response_text)
    if not title:
        logger.debug("No primary title found in response — skipping recommendations")
        return ""

    # Step 2: look up its rating/meta score
    scores = _lookup_movie_scores(title)
    if not scores:
        logger.debug(f"No scores found for '{title}' — skipping recommendations")
        return ""

    logger.info(
        f"Recommendations for '{scores['title']}' "
        f"(IMDB={scores['imdb_rating']}, Meta={scores['meta_score']})"
    )

    # Step 3: find similar movies
    similar = get_similar_movies(
        imdb_rating=scores["imdb_rating"],
        meta_score=scores["meta_score"],
        exclude_title=scores["title"],
    )

    if not similar:
        return ""

    # Step 4: format the recommendation block
    lines = [
        f"\n\n---",
        f"**You might also enjoy** _(similar IMDB rating ~{scores['imdb_rating']}, "
        f"Meta score ~{int(scores['meta_score'])})_:",
    ]
    for movie in similar:
        lines.append(
            f"- **{movie['title']}** ({movie['year']}) — "
            f"IMDB {movie['imdb_rating']} | Meta {int(movie['meta_score'])} | "
            f"{movie['genre']}"
        )

    return "\n".join(lines)
