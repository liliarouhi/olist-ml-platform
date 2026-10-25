-- 00_staging.sql
-- Raw copies of the Olist CSV files. 

DROP SCHEMA IF EXISTS staging CASCADE;
CREATE SCHEMA staging;

CREATE TABLE staging.customers (customer_id text, customer_unique_id text, customer_zip_code_prefix text, customer_city text, customer_state text);
CREATE TABLE staging.geolocation (geolocation_zip_code_prefix text, geolocation_lat text, geolocation_lng text, geolocation_city text, geolocation_state text);
CREATE TABLE staging.orders (order_id text, customer_id text, order_status text, order_purchase_timestamp text, order_approved_at text, order_delivered_carrier_date text, order_delivered_customer_date text, order_estimated_delivery_date text);
CREATE TABLE staging.order_items (order_id text, order_item_id text, product_id text, seller_id text, shipping_limit_date text, price text, freight_value text);
CREATE TABLE staging.order_payments (order_id text, payment_sequential text, payment_type text, payment_installments text, payment_value text);
CREATE TABLE staging.order_reviews (review_id text, order_id text, review_score text, review_comment_title text, review_comment_message text, review_creation_date text, review_answer_timestamp text);
CREATE TABLE staging.products (product_id text, product_category_name text, product_name_lenght text, product_description_lenght text, product_photos_qty text, product_weight_g text, product_length_cm text, product_height_cm text, product_width_cm text);
CREATE TABLE staging.sellers (seller_id text, seller_zip_code_prefix text, seller_city text, seller_state text);
CREATE TABLE staging.category_translation (product_category_name text, product_category_name_english text);

\copy staging.customers FROM 'data/raw/olist_customers_dataset.csv' WITH (FORMAT csv, HEADER true)
\copy staging.geolocation FROM 'data/raw/olist_geolocation_dataset.csv' WITH (FORMAT csv, HEADER true)
\copy staging.orders FROM 'data/raw/olist_orders_dataset.csv' WITH (FORMAT csv, HEADER true)
\copy staging.order_items FROM 'data/raw/olist_order_items_dataset.csv' WITH (FORMAT csv, HEADER true)
\copy staging.order_payments FROM 'data/raw/olist_order_payments_dataset.csv' WITH (FORMAT csv, HEADER true)
\copy staging.order_reviews FROM 'data/raw/olist_order_reviews_dataset.csv' WITH (FORMAT csv, HEADER true)
\copy staging.products FROM 'data/raw/olist_products_dataset.csv' WITH (FORMAT csv, HEADER true)
\copy staging.sellers FROM 'data/raw/olist_sellers_dataset.csv' WITH (FORMAT csv, HEADER true)
\copy staging.category_translation FROM 'data/raw/product_category_name_translation.csv' WITH (FORMAT csv, HEADER true)