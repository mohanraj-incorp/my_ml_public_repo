"""
Helper to interactively annotate golden dataset entries.

Run with:
    python scripts/seed_golden_dataset.py

For each query in QUERIES_TO_ANNOTATE, this script:
  1. Runs the query through the live agent
  2. Prints the agent's response
  3. Asks you to confirm or edit it as the ground truth
  4. Saves to evaluation/golden_dataset.json

Use this to build up the golden dataset incrementally rather than writing
all expected answers manually from memory.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import process_query
from rag.indexer import build_and_save_indexes
from tools.sqlite_tools import init_sqlite_db
from memory.long_term import init_preferences_table

GOLDEN_PATH = "evaluation/golden_dataset.json"

# Add new queries here to annotate them interactively
QUERIES_TO_ANNOTATE = [
    "When was The Godfather released?",
    "Find movies with time travel in the plot",
    "Which animated movies are in the top 1000?",
]


def load_existing() -> list[dict]:
    if os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH) as f:
            return json.load(f)
    return []


def save(dataset: list[dict]) -> None:
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


async def annotate():
    build_and_save_indexes()
    init_sqlite_db()
    await init_preferences_table()

    dataset = load_existing()
    existing_queries = {entry["query"] for entry in dataset}

    for query in QUERIES_TO_ANNOTATE:
        if query in existing_queries:
            print(f"\n[SKIP] Already in dataset: {query}")
            continue

        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("Running agent…")

        session_id = f"seed_{len(dataset)}"
        response = await process_query(query, session_id)

        print(f"\nAgent response:\n{response}")

        print("\nOptions: [Enter] Accept as-is | [e] Edit | [s] Skip")
        choice = input("> ").strip().lower()

        if choice == "s":
            print("Skipped.")
            continue

        ground_truth = response
        if choice == "e":
            print("Enter corrected ground truth (end with a blank line):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            ground_truth = "\n".join(lines)

        entry_id = f"SEED_{len(dataset) + 1:03d}"
        route = input("Expected route [analytical/semantic/clarify]: ").strip() or "semantic"
        query_type = input("Query type [A/B/C/D]: ").strip().upper() or "B"

        entry = {
            "id": entry_id,
            "type": query_type,
            "category": "seeded",
            "query": query,
            "expected_answer": ground_truth,
            "expected_route": route,
            "expected_context": [],
            "notes": "Seeded via seed_golden_dataset.py",
        }
        dataset.append(entry)
        save(dataset)
        print(f"Saved entry {entry_id}.")

    print(f"\nDone. Dataset now has {len(dataset)} entries at {GOLDEN_PATH}")


if __name__ == "__main__":
    asyncio.run(annotate())
