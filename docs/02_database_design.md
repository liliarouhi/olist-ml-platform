# 02 — Database Design

## 1. Approach
Raw CSV files are loaded unchanged into a `staging` schema (all columns as text),
so loading never fails on dirty data. Before designing the normalized `core` schema,
every candidate key and functional dependency (FD) is verified with SQL on the real data.

FD test used: `A → B` holds iff
`SELECT A FROM t GROUP BY A HAVING count(DISTINCT B) > 1` returns no rows.

## 2. Evidence

| Test | Result | Conclusion |
|---|---|---|
| `customer_id` unique | 99,441 / 99,441 | candidate key |
| `customer_unique_id` unique | 96,096 distinct of 99,441 | one person has many `customer_id`s |
| `customer_unique_id → zip_code_prefix` | 250 violations | address belongs to `customer_id`, not to the person |
| `review_id` unique | 98,410 / 99,224 | not a key |
| `(review_id, order_id)` unique | 99,224 / 99,224 | composite key |
| distinct `order_id` in reviews | 98,673 of 99,224 rows | an order can have several reviews: many-to-many |
| `zip_code_prefix → city` (geolocation) | 8,556 of 19,015 prefixes violate | not an FD (spelling variants and shared prefixes) |
| `(order_id, order_item_id)` unique | 112,650 / 112,650 | composite key |
| `review_id → (score, message)` | TBD | decides whether reviews are decomposed |