-- 02_indexes.sql
-- Run after loading data (bulk load is faster without indexes). Evidence: docs/03_indexing_benchmarks.md

-- Remove experimental / redundant indexes
DROP INDEX IF EXISTS core.idx_order_items_seller_id;
DROP INDEX IF EXISTS core.idx_order_items_seller_shipping;
DROP INDEX IF EXISTS core.idx_order_items_seller_cover;

-- One index serves: filter by seller, sort by shipping date, and index-only reads of price
CREATE INDEX idx_order_items_seller ON core.order_items (seller_id, shipping_limit_at) INCLUDE (price);

-- Foreign keys are not indexed automatically in PostgreSQL
CREATE INDEX idx_order_items_product        ON core.order_items (product_id);
CREATE INDEX idx_customer_accounts_person   ON core.customer_accounts (customer_unique_id);
CREATE INDEX idx_order_reviews_order        ON core.order_reviews (order_id);
CREATE INDEX idx_products_category          ON core.products (category_name);

-- Time-range queries for analytics
CREATE INDEX idx_orders_purchased_at        ON core.orders (purchased_at);

VACUUM ANALYZE;
