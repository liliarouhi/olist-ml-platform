-- 04_star_schema.sql
-- OLAP layer: a star schema built from the normalized core schema.
-- Fact grain: one row per order item. Fact table is range-partitioned by purchase year.

DROP SCHEMA IF EXISTS dw CASCADE;
CREATE SCHEMA dw;

-- Date dimension: one row per day, "smart" key YYYYMMDD
CREATE TABLE dw.dim_date (
    date_key    int PRIMARY KEY,
    full_date   date NOT NULL UNIQUE,
    year        smallint NOT NULL,
    quarter     smallint NOT NULL,
    month       smallint NOT NULL,
    month_start date NOT NULL,
    day_of_week smallint NOT NULL,   -- 1 = Monday ... 7 = Sunday
    is_weekend  boolean NOT NULL
);

INSERT INTO dw.dim_date
SELECT to_char(d, 'YYYYMMDD')::int, d::date, extract(year FROM d), extract(quarter FROM d),
       extract(month FROM d), date_trunc('month', d)::date,
       extract(isodow FROM d), extract(isodow FROM d) IN (6, 7)
FROM generate_series('2016-01-01'::date, '2018-12-31'::date, interval '1 day') AS g(d);

-- Customer dimension: denormalized on purpose (address and coordinates in one row)
CREATE TABLE dw.dim_customer (
    customer_key       int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id        text NOT NULL UNIQUE,
    customer_unique_id text NOT NULL,
    city               text NOT NULL,
    state              char(2) NOT NULL,
    lat                double precision,
    lng                double precision
);

INSERT INTO dw.dim_customer (customer_id, customer_unique_id, city, state, lat, lng)
SELECT ca.customer_id, ca.customer_unique_id, ca.city, ca.state, z.lat, z.lng
FROM core.customer_accounts ca
JOIN core.zip_codes z USING (zip_code_prefix);

CREATE TABLE dw.dim_seller (
    seller_key int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    seller_id  text NOT NULL UNIQUE,
    city       text NOT NULL,
    state      char(2) NOT NULL,
    lat        double precision,
    lng        double precision
);

INSERT INTO dw.dim_seller (seller_id, city, state, lat, lng)
SELECT s.seller_id, s.city, s.state, z.lat, z.lng
FROM core.sellers s
JOIN core.zip_codes z USING (zip_code_prefix);

CREATE TABLE dw.dim_product (
    product_key int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_id  text NOT NULL UNIQUE,
    category    text NOT NULL,
    weight_g    int,
    volume_cm3  int,
    photos_qty  int
);

INSERT INTO dw.dim_product (product_id, category, weight_g, volume_cm3, photos_qty)
SELECT p.product_id,
       coalesce(c.category_name_english, p.category_name, 'unknown'),
       p.weight_g, p.length_cm * p.height_cm * p.width_cm, p.photos_qty
FROM core.products p
LEFT JOIN core.categories c USING (category_name);

-- Fact table, partitioned by purchase date (the partition key must be part of the primary key)
CREATE TABLE dw.fact_order_items (
    order_id       text NOT NULL,
    order_item_id  smallint NOT NULL,
    purchased_date date NOT NULL,
    date_key       int NOT NULL REFERENCES dw.dim_date,
    customer_key   int NOT NULL REFERENCES dw.dim_customer,
    product_key    int NOT NULL REFERENCES dw.dim_product,
    seller_key     int NOT NULL REFERENCES dw.dim_seller,
    order_status   text NOT NULL,
    price          numeric(10,2) NOT NULL,
    freight_value  numeric(10,2) NOT NULL,
    delivery_days  int,           -- NULL when not delivered
    is_late        boolean,       -- NULL when not delivered
    review_score   numeric(3,2),  -- average score of the order's reviews
    PRIMARY KEY (order_id, order_item_id, purchased_date)
) PARTITION BY RANGE (purchased_date);

CREATE TABLE dw.fact_order_items_2016 PARTITION OF dw.fact_order_items FOR VALUES FROM ('2016-01-01') TO ('2017-01-01');
CREATE TABLE dw.fact_order_items_2017 PARTITION OF dw.fact_order_items FOR VALUES FROM ('2017-01-01') TO ('2018-01-01');
CREATE TABLE dw.fact_order_items_2018 PARTITION OF dw.fact_order_items FOR VALUES FROM ('2018-01-01') TO ('2019-01-01');
CREATE TABLE dw.fact_order_items_default PARTITION OF dw.fact_order_items DEFAULT;

INSERT INTO dw.fact_order_items (order_id, order_item_id, purchased_date, date_key, customer_key,
                                 product_key, seller_key, order_status, price, freight_value,
                                 delivery_days, is_late, review_score)
SELECT oi.order_id, oi.order_item_id, o.purchased_at::date,
       to_char(o.purchased_at, 'YYYYMMDD')::int,
       dc.customer_key, dp.product_key, ds.seller_key,
       o.order_status, oi.price, oi.freight_value,
       o.delivered_customer_at::date - o.purchased_at::date,
       o.delivered_customer_at::date > o.estimated_delivery_date,
       r.avg_score
FROM core.order_items oi
JOIN core.orders o        USING (order_id)
JOIN dw.dim_customer dc   ON dc.customer_id = o.customer_id
JOIN dw.dim_product dp    ON dp.product_id = oi.product_id
JOIN dw.dim_seller ds     ON ds.seller_id = oi.seller_id
LEFT JOIN (
    SELECT orv.order_id, avg(rv.score)::numeric(3,2) AS avg_score
    FROM core.order_reviews orv
    JOIN core.reviews rv USING (review_id)
    GROUP BY orv.order_id
) r ON r.order_id = oi.order_id;

-- Indexes on a partitioned table are created on every partition automatically
CREATE INDEX ON dw.fact_order_items (date_key);
CREATE INDEX ON dw.fact_order_items (product_key);
CREATE INDEX ON dw.fact_order_items (customer_key);

ANALYZE dw.dim_date, dw.dim_customer, dw.dim_seller, dw.dim_product, dw.fact_order_items;