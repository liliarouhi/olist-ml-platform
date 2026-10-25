-- 01_schema.sql

DROP SCHEMA IF EXISTS core CASCADE;
CREATE SCHEMA core;

CREATE TABLE core.zip_codes (
    zip_code_prefix text PRIMARY KEY CHECK (zip_code_prefix ~ '^[0-9]{5}$'),
    lat double precision CHECK (lat BETWEEN -90 AND 90),
    lng double precision CHECK (lng BETWEEN -180 AND 180)
);

CREATE TABLE core.customers (
    customer_unique_id text PRIMARY KEY
);

CREATE TABLE core.customer_accounts (
    customer_id        text PRIMARY KEY,
    customer_unique_id text NOT NULL REFERENCES core.customers,
    zip_code_prefix    text NOT NULL REFERENCES core.zip_codes,
    city               text NOT NULL,
    state              char(2) NOT NULL
);

CREATE TABLE core.sellers (
    seller_id       text PRIMARY KEY,
    zip_code_prefix text NOT NULL REFERENCES core.zip_codes,
    city            text NOT NULL,
    state           char(2) NOT NULL
);

CREATE TABLE core.categories (
    category_name         text PRIMARY KEY,
    category_name_english text
);

CREATE TABLE core.products (
    product_id         text PRIMARY KEY,
    category_name      text REFERENCES core.categories,
    name_length        int CHECK (name_length >= 0),
    description_length int CHECK (description_length >= 0),
    photos_qty         int CHECK (photos_qty >= 0),
    weight_g           int CHECK (weight_g >= 0),
    length_cm          int CHECK (length_cm >= 0),
    height_cm          int CHECK (height_cm >= 0),
    width_cm           int CHECK (width_cm >= 0)
);

CREATE TABLE core.orders (
    order_id                text PRIMARY KEY,
    customer_id             text NOT NULL UNIQUE REFERENCES core.customer_accounts,
    order_status            text NOT NULL CHECK (order_status IN
        ('created','approved','invoiced','processing','shipped','delivered','canceled','unavailable')),
    purchased_at            timestamp NOT NULL,
    approved_at             timestamp,
    delivered_carrier_at    timestamp,
    delivered_customer_at   timestamp,
    estimated_delivery_date date NOT NULL
);

CREATE TABLE core.order_items (
    order_id          text REFERENCES core.orders ON DELETE CASCADE,
    order_item_id     smallint CHECK (order_item_id >= 1),
    product_id        text NOT NULL REFERENCES core.products,
    seller_id         text NOT NULL REFERENCES core.sellers,
    shipping_limit_at timestamp NOT NULL,
    price             numeric(10,2) NOT NULL CHECK (price >= 0),
    freight_value     numeric(10,2) NOT NULL CHECK (freight_value >= 0),
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE core.payments (
    order_id           text REFERENCES core.orders ON DELETE CASCADE,
    payment_sequential smallint CHECK (payment_sequential >= 1),
    payment_type       text NOT NULL CHECK (payment_type IN
        ('credit_card','boleto','voucher','debit_card','not_defined')),
    installments       smallint NOT NULL CHECK (installments >= 0),
    amount             numeric(10,2) NOT NULL CHECK (amount >= 0),
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE core.reviews (
    review_id   text PRIMARY KEY,
    score       smallint NOT NULL CHECK (score BETWEEN 1 AND 5),
    title       text,
    message     text,
    created_at  timestamp NOT NULL,
    answered_at timestamp
);

CREATE TABLE core.order_reviews (
    review_id text REFERENCES core.reviews ON DELETE CASCADE,
    order_id  text REFERENCES core.orders ON DELETE CASCADE,
    PRIMARY KEY (review_id, order_id)
);