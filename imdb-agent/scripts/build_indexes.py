"""
One-time script to build BM25 and FAISS indexes from the IMDB CSV.

Run before starting the app for the first time:
    python scripts/build_indexes.py

Or force a rebuild after updating the dataset:
    python scripts/build_indexes.py --force

The app (app/main.py) also calls build_and_save_indexes() on startup but
running this separately gives you a clean log of the indexing process
and avoids first-request latency.
"""
import argparse
import logging
import sys
import os

# Allow running from project root: python scripts/build_indexes.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.indexer import build_and_save_indexes
from tools.sqlite_tools import init_sqlite_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Build IMDB search indexes")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild indexes even if they already exist",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Override path to the IMDB CSV file",
    )
    args = parser.parse_args()

    print("Step 1/2: Building BM25 + FAISS indexes…")
    build_and_save_indexes(csv_path=args.csv, force_rebuild=args.force)
    print("  Done.")

    print("Step 2/2: Initialising SQLite database…")
    init_sqlite_db(csv_path=args.csv)
    print("  Done.")

    print("\nAll indexes ready. Start the app with: streamlit run app/main.py")


if __name__ == "__main__":
    main()
