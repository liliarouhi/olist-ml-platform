# 03 — Indexing and Query Optimization

Query: items of one seller (45 of 112,650 rows), measured with `EXPLAIN (ANALYZE, BUFFERS)`.

| Query | Index | Plan | Pages read | Time |
|---|---|---|---|---|
| `WHERE seller_id = ?` | none | Seq Scan | 2,167 | 60.1 ms |
| `WHERE seller_id = ?` | `(seller_id)` | Bitmap Index + Heap Scan | 46 | 3.1 ms |
| `... ORDER BY shipping_limit_at DESC LIMIT 10` | `(seller_id)` | Bitmap scan + top-N Sort | 46 | 3.0 ms |
| `... ORDER BY shipping_limit_at DESC LIMIT 10` | `(seller_id, shipping_limit_at)` | Index Scan Backward, no Sort | 13 | 0.1 ms |
| `SELECT price ... WHERE seller_id = ?` | `(seller_id) INCLUDE (price)` | Index Only Scan, 0 heap fetches | 5 | 0.9 ms |

## Findings
- PostgreSQL indexes primary keys and UNIQUE columns automatically, but not foreign keys.
- A composite index ordered as (equality column, sort column) removes the Sort node and lets LIMIT stop early.
- A covering index (`INCLUDE`) enables index-only scans; `VACUUM` updates the visibility map so no heap fetches are needed.
- Page counts (Buffers) are a more stable comparison than timings, which vary with caching.
- The three experimental indexes were merged into one: `(seller_id, shipping_limit_at) INCLUDE (price)`, since redundant indexes slow down writes.

Final indexes: `sql/02_indexes.sql`.


## Scaling benchmark

Script: `scripts/benchmark_indexes.py`. The order items table is replicated 1, 3, 9 and 27 times;
each copy gets its own seller ids, so a seller always has 45 rows while the table grows.
Each query runs once to warm the cache, then 5 measured runs; the median is reported.
Raw data: `docs/figures/index_benchmark.csv`.

![Index scaling benchmark](figures/index_benchmark.png)

| Rows | Seq Scan time | Seq Scan pages | Index time | Index pages |
|---:|---:|---:|---:|---:|
| 112,650 | 5.80 ms | 2,304 | 0.016 ms | 47 |
| 337,950 | 11.73 ms | 6,912 | 0.014 ms | 47 |
| 1,013,850 | 24.54 ms | 20,736 | 0.016 ms | 47 |
| 3,041,550 | 71.37 ms | 62,080 | 0.015 ms | 47 |

### Findings
- Pages read by the Seq Scan grow exactly linearly with table size (O(n)).
- The index reads 47 pages at every size: B-tree depth grows logarithmically, and at this
  scale it did not change at all.
- At 3 million rows the index lookup is about 4,700× faster.
- Seq Scan time grew less than linearly (27× rows, 12× time) because PostgreSQL switches to a
  Parallel Seq Scan on larger tables: the same amount of work is split across CPU cores.
- The 112k-row Seq Scan took 60 ms in the earlier single cold run and 5.8 ms here with a
  warm cache, which is why page counts are the more reliable metric.


  ## Advanced SQL: analytical queries

File: `sql/03_advanced_queries.sql`. Full output: `docs/figures/advanced_queries_output.txt`.

| Query | Technique | Key result |
|---|---|---|
| Q1 Monthly revenue, MoM growth, running total | CTE, `LAG`, running `SUM OVER` | 15.42M BRL delivered revenue; peak Nov 2017 (+53.6%, Black Friday) |
| Q2 Top 3 sellers per state | `RANK() OVER (PARTITION BY ...)` | sellers exist in only 23 states; several states have 1–2 active sellers |
| Q3 Repeat customers | `LAG` per customer, `FILTER` | 2,997 of 96,096 customers (3.1%) ordered again; 78.2 days between orders on average |
| Q4 Late deliveries per state | `percentile_cont` (median) | late rate from 2.8% (AM) to 21.4% (AL); SP 4.5% with a 7-day median |
| Q5 Days without orders | recursive CTE calendar | 140 empty days, all before 2017-01-05 or after 2018-08-31 |

### Findings
- **Valid analysis window:** every day from 2017-01-05 to 2018-08-31 has orders; the edges of the
  dataset are incomplete. Models and dashboards use January 2017 to August 2018.
- **`LAG` compares rows, not calendar months:** November 2016 has no orders, so December 2016 was
  compared with October. Gap-free time series need a join with a calendar (the Q5 technique).
- **Repeat purchases are rare (3.1%),** so customer-history features will carry little signal.
- **Lateness depends on the estimate, not on distance:** remote states (AM, AP, RR) have the longest
  median delivery (25–26 days) but low late rates, while AL and MA have shorter deliveries and the
  highest late rates. This motivates the ML target: predicting whether an order arrives after its
  estimated date.