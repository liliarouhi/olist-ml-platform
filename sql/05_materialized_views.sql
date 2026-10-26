-- 05_materialized_views.sql
-- Precomputed aggregates for the dashboard. After each data load:
--   REFRESH MATERIALIZED VIEW CONCURRENTLY dw.mv_monthly_category_revenue;
-- CONCURRENTLY keeps the view readable during the refresh and requires a unique index.

DROP MATERIALIZED VIEW IF EXISTS dw.mv_monthly_category_revenue, dw.mv_state_delivery, dw.mv_revenue_rollup;

-- Valid analysis window (docs/03): 2017-01 to 2018-08. Filtering on the partition key enables pruning.
CREATE MATERIALIZED VIEW dw.mv_monthly_category_revenue AS
SELECT d.month_start, p.category,
       count(DISTINCT f.order_id) AS orders,
       count(*)                   AS items,
       sum(f.price)               AS revenue
FROM dw.fact_order_items f
JOIN dw.dim_date d    USING (date_key)
JOIN dw.dim_product p USING (product_key)
WHERE f.order_status = 'delivered'
  AND f.purchased_date BETWEEN '2017-01-01' AND '2018-08-31'
GROUP BY 1, 2;
CREATE UNIQUE INDEX ON dw.mv_monthly_category_revenue (month_start, category);

-- Order-level delivery quality per customer state (one row per order, not per item)
CREATE MATERIALIZED VIEW dw.mv_state_delivery AS
WITH delivered_orders AS (
    SELECT DISTINCT ON (f.order_id) f.order_id, c.state, f.is_late, f.delivery_days, f.review_score
    FROM dw.fact_order_items f
    JOIN dw.dim_customer c USING (customer_key)
    WHERE f.order_status = 'delivered' AND f.delivery_days IS NOT NULL
)
SELECT state,
       count(*)                                AS delivered_orders,
       round(100.0 * avg(is_late::int), 1)     AS late_pct,
       round(avg(delivery_days), 1)            AS avg_delivery_days,
       round(avg(review_score), 2)             AS avg_review_score
FROM delivered_orders
GROUP BY state;
CREATE UNIQUE INDEX ON dw.mv_state_delivery (state);

-- Revenue with subtotals: months, quarters, years and a grand total in one query.
-- grouping_level: 0 = month, 1 = quarter subtotal, 3 = year subtotal, 7 = grand total
CREATE MATERIALIZED VIEW dw.mv_revenue_rollup AS
SELECT d.year, d.quarter, d.month,
       GROUPING(d.year, d.quarter, d.month) AS grouping_level,
       sum(f.price)                         AS revenue,
       count(DISTINCT f.order_id)           AS orders
FROM dw.fact_order_items f
JOIN dw.dim_date d USING (date_key)
WHERE f.order_status = 'delivered'
GROUP BY ROLLUP (d.year, d.quarter, d.month);