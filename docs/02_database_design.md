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

## 3. Core schema design

| Table | Primary key | Notes |
|---|---|---|
| `zip_codes` | `zip_code_prefix` | UNION of all prefixes; avg lat/lng from geolocation; NULL when unknown |
| `customers` | `customer_unique_id` | the real person |
| `customer_accounts` | `customer_id` | address lives here (FD on person does not hold) |
| `sellers` | `seller_id` | |
| `categories` | `category_name` | UNION with product categories; English name may be NULL |
| `products` | `product_id` | misspelled source columns renamed (`lenght` → `length`) |
| `orders` | `order_id` | `customer_id` UNIQUE: one-to-one with `customer_accounts` |
| `order_items` | `(order_id, order_item_id)` | money as `numeric(10,2)`, never float |
| `payments` | `(order_id, payment_sequential)` | |
| `reviews` | `review_id` | BCNF decomposition, since `review_id → (score, text, dates)` holds |
| `order_reviews` | `(review_id, order_id)` | many-to-many bridge table |

ER diagram: `docs/figures/er_diagram.png`

## 4. ETL (staging → core)

Implemented in `src/olist_ml/etl/loader.py` (`CoreLoader`). Steps run in foreign-key order
inside a single transaction, so a failure leaves `core` unchanged. Tables are truncated first,
which makes the load idempotent.

Cleaning applied: empty strings → NULL (`NULLIF`), explicit type casts, city names normalized
with `lower(unaccent(trim(...)))`.

| Result | Value |
|---|---|
| Rows loaded | 764,444 in 11 tables (~8 s) |
| Zip prefixes without coordinates | 162 of 19,177 (present in customers/sellers, missing from geolocation) |
| Categories without English name | 2 (`pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos`) |
| Distinct customer cities raw → cleaned | 4,119 → 4,119 |

Finding: customer and seller city names were already normalized in the source. The accent
variants (e.g. `sao paulo` / `são paulo`) exist only in the geolocation file, which the core
schema reduces to coordinates per zip prefix, so they do not reach the core tables.
The normalization step is kept as a safeguard for future data.

Without the UNION approach, 162 customer/seller rows would have violated the foreign key to
`zip_codes`, and products in the 2 untranslated categories would have violated the foreign key
to `categories`.