-- 06_features.sql
-- Model dataset: one row per delivered order, features known at purchase time, target is_late.
-- Leakage rule: nothing that happens after the purchase may become a feature.

DROP SCHEMA IF EXISTS ml CASCADE;
CREATE SCHEMA ml;

CREATE TABLE ml.order_features AS
WITH base AS (   -- delivered orders inside the valid window (docs/03)
    SELECT order_id, customer_id, purchased_at, estimated_delivery_date, delivered_customer_at
    FROM core.orders
    WHERE order_status = 'delivered'
      AND delivered_customer_at IS NOT NULL
      AND purchased_at >= '2017-01-01' AND purchased_at < '2018-09-01'
),
items AS (
    SELECT oi.order_id,
           count(*)                                            AS n_items,
           count(DISTINCT oi.seller_id)                        AS n_sellers,
           sum(oi.price)::float                                AS total_price,
           sum(oi.freight_value)::float                        AS total_freight,
           sum(p.weight_g)::float                              AS total_weight_g,
           sum(p.length_cm * p.height_cm * p.width_cm)::float  AS total_volume_cm3,
           max(p.photos_qty)                                   AS max_photos
    FROM core.order_items oi
    JOIN core.products p USING (product_id)
    GROUP BY oi.order_id
),
main_item AS (   -- the most expensive item decides the order's main category and seller
    SELECT DISTINCT ON (oi.order_id)
           oi.order_id, oi.seller_id,
           coalesce(c.category_name_english, p.category_name, 'unknown') AS main_category
    FROM core.order_items oi
    JOIN core.products p USING (product_id)
    LEFT JOIN core.categories c USING (category_name)
    ORDER BY oi.order_id, oi.price DESC
),
payments AS (
    SELECT order_id,
           (array_agg(payment_type ORDER BY amount DESC))[1] AS payment_type,
           max(installments)                                AS installments
    FROM core.payments
    GROUP BY order_id
),
-- Seller history without leakage: one event stream per seller, sorted by time.
-- Purchase events (is_delivery = 0) only count deliveries that finished before them.
seller_events AS (
    SELECT m.seller_id, b.order_id, b.purchased_at AS event_time, 0 AS is_delivery, 0 AS late
    FROM base b JOIN main_item m USING (order_id)
    UNION ALL
    SELECT m.seller_id, NULL, o.delivered_customer_at, 1,
           (o.delivered_customer_at::date > o.estimated_delivery_date)::int
    FROM core.orders o JOIN main_item m USING (order_id)
    WHERE o.order_status = 'delivered' AND o.delivered_customer_at IS NOT NULL
),
seller_history AS (
    SELECT order_id, prior_deliveries, prior_late
    FROM (
        SELECT order_id, is_delivery,
               sum(is_delivery) OVER w AS prior_deliveries,
               sum(late)        OVER w AS prior_late
        FROM seller_events
        WINDOW w AS (PARTITION BY seller_id ORDER BY event_time, is_delivery
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ) s
    WHERE is_delivery = 0
)
SELECT b.order_id,
       b.purchased_at,
       (b.delivered_customer_at::date > b.estimated_delivery_date)       AS is_late,      -- target
       extract(month  FROM b.purchased_at)::int                          AS purchase_month,
       extract(isodow FROM b.purchased_at)::int                          AS purchase_weekday,
       extract(hour   FROM b.purchased_at)::int                          AS purchase_hour,
       b.estimated_delivery_date - b.purchased_at::date                  AS estimated_days,
       i.n_items, i.n_sellers, i.total_price, i.total_freight,
       i.total_freight / nullif(i.total_price, 0)                        AS freight_ratio,
       i.total_weight_g, i.total_volume_cm3, i.max_photos,
       m.main_category,
       pay.payment_type, pay.installments,
       ca.state                                                          AS customer_state,
       s.state                                                           AS seller_state,
       ca.state = s.state                                                AS same_state,
       2 * 6371 * asin(least(1, sqrt(
           power(sin(radians(zs.lat - zc.lat) / 2), 2)
           + cos(radians(zc.lat)) * cos(radians(zs.lat)) * power(sin(radians(zs.lng - zc.lng) / 2), 2)
       )))                                                               AS distance_km,
       sh.prior_deliveries                                               AS seller_prior_deliveries,
       sh.prior_late                                                     AS seller_prior_late
FROM base b
JOIN items i               USING (order_id)
JOIN main_item m           USING (order_id)
LEFT JOIN payments pay     USING (order_id)
LEFT JOIN seller_history sh USING (order_id)
JOIN core.customer_accounts ca ON ca.customer_id = b.customer_id
JOIN core.zip_codes zc     ON zc.zip_code_prefix = ca.zip_code_prefix
JOIN core.sellers s        ON s.seller_id = m.seller_id
JOIN core.zip_codes zs     ON zs.zip_code_prefix = s.zip_code_prefix;

ALTER TABLE ml.order_features ADD PRIMARY KEY (order_id);
ANALYZE ml.order_features;