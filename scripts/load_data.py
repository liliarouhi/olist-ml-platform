"""Rebuild the core schema from staging. Usage: python scripts/load_data.py"""
import logging

from olist_ml.db.connection import Database
from olist_ml.etl.loader import CoreLoader


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with Database() as db:
        counts = CoreLoader(db).run()
    logging.info("Committed: %d tables, %d rows in total.", len(counts), sum(counts.values()))


if __name__ == "__main__":
    main()