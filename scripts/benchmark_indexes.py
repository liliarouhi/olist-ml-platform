"""Benchmark a B-tree index as the table grows (Seq Scan vs index lookup).

Usage:  python scripts/benchmark_indexes.py
Output: docs/figures/index_benchmark.csv and docs/figures/index_benchmark.png
"""
from __future__ import annotations

import csv
import logging
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from psycopg import sql

from olist_ml.db.connection import Database
from olist_ml.viz.plots import plot_index_benchmark

logger = logging.getLogger(__name__)

COPIES = [1, 3, 9, 27]  
REPEATS = 5
OUT_DIR = Path("docs/figures")


@dataclass(frozen=True)
class BenchmarkResult:
    rows: int
    indexed: bool
    median_ms: float
    pages: int


class IndexBenchmark:
    """Builds bench.items at several sizes and times one seller lookup with and without an index.

    Each copy of the source data gets its own seller ids, so rows per seller stay constant
    while the table grows, like a marketplace gaining new sellers.
    """

    def __init__(self, db: Database, copies: list[int], repeats: int = REPEATS) -> None:
        self._db = db
        self._copies = copies
        self._repeats = repeats

    def run(self) -> list[BenchmarkResult]:
        self._db.execute("CREATE SCHEMA IF NOT EXISTS bench")
        seller = self._pick_seller()
        results: list[BenchmarkResult] = []
        try:
            for copies in self._copies:
                rows = self._build_table(copies)
                results.append(self._measure(rows, seller, indexed=False))
                self._db.execute("CREATE INDEX ON bench.items (seller_id)")
                self._db.execute("VACUUM ANALYZE bench.items")
                results.append(self._measure(rows, seller, indexed=True))
        finally:
            self._db.execute("DROP SCHEMA IF EXISTS bench CASCADE") 
        return results

    def _pick_seller(self) -> str:
        row = self._db.fetch_one(
            "SELECT seller_id FROM core.order_items GROUP BY 1 ORDER BY count(*) DESC OFFSET 500 LIMIT 1"
        )
        return f"{row[0]}-1" 

    def _build_table(self, copies: int) -> int:
        self._db.execute("DROP TABLE IF EXISTS bench.items")
        self._db.execute(sql.SQL("""
            CREATE TABLE bench.items AS
            SELECT oi.order_id || '-' || g AS order_id, oi.order_item_id, oi.product_id,
                   oi.seller_id || '-' || g AS seller_id,
                   oi.shipping_limit_at, oi.price, oi.freight_value
            FROM core.order_items oi
            CROSS JOIN generate_series(1, {copies}) AS g
        """).format(copies=sql.Literal(copies)))
        self._db.execute("VACUUM ANALYZE bench.items")
        return self._db.fetch_one("SELECT count(*) FROM bench.items")[0]

    def _measure(self, rows: int, seller: str, indexed: bool) -> BenchmarkResult:
        query = sql.SQL(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT * FROM bench.items WHERE seller_id = {}"
        ).format(sql.Literal(seller))
        times: list[float] = []
        pages = 0
        for attempt in range(self._repeats + 1):
            plan = self._db.fetch_one(query)[0][0]
            if attempt == 0:
                continue  
            times.append(plan["Execution Time"])
            top = plan["Plan"]
            pages = top["Shared Hit Blocks"] + top["Shared Read Blocks"]
        result = BenchmarkResult(rows, indexed, round(statistics.median(times), 3), pages)
        logger.info("%9d rows  index=%-5s  %9.3f ms  %7d pages",
                    rows, indexed, result.median_ms, result.pages)
        return result


def save_csv(results: list[BenchmarkResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with Database(autocommit=True) as db:
        results = IndexBenchmark(db, COPIES).run()
    save_csv(results, OUT_DIR / "index_benchmark.csv")
    plot_index_benchmark(results, OUT_DIR / "index_benchmark.png")
    logging.info("Saved results and figure to %s", OUT_DIR)


if __name__ == "__main__":
    main()