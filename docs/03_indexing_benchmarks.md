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