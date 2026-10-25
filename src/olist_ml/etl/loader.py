"""ETL: transform data from the staging schema and load it into the core schema."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from olist_ml.db.connection import Database

logger = logging.getLogger(__name__)

CLEAN_CITY = "lower(unaccent(trim({col})))"


@dataclass(frozen=True)
class TransformStep:
    
    target: str
    sql: str


STEPS: list[TransformStep] = [
    TransformStep("core.zip_codes", """
        INSERT INTO core.zip_codes (zip_code_prefix, lat, lng)
        SELECT z.zip, g.lat, g.lng
        FROM (
            SELECT customer_zip_code_prefix AS zip FROM staging.customers
            UNION SELECT seller_zip_code_prefix FROM staging.sellers
            UNION SELECT geolocation_zip_code_prefix FROM staging.geolocation
        ) z
        LEFT JOIN (
            SELECT geolocation_zip_code_prefix AS zip,
                   avg(geolocation_lat::double precision) AS lat,
                   avg(geolocation_lng::double precision) AS lng
            FROM staging.geolocation
            GROUP BY 1
        ) g USING (zip)
    """),
    TransformStep("core.customers", """
        INSERT INTO core.customers (customer_unique_id)
        SELECT DISTINCT customer_unique_id FROM staging.customers
    """),
    TransformStep("core.customer_accounts", f"""
        INSERT INTO core.customer_accounts (customer_id, customer_unique_id, zip_code_prefix, city, state)
        SELECT customer_id, customer_unique_id, customer_zip_code_prefix,
               {CLEAN_CITY.format(col="customer_city")}, upper(customer_state)
        FROM staging.customers
    """),
    TransformStep("core.sellers", f"""
        INSERT INTO core.sellers (seller_id, zip_code_prefix, city, state)
        SELECT seller_id, seller_zip_code_prefix,
               {CLEAN_CITY.format(col="seller_city")}, upper(seller_state)
        FROM staging.sellers
    """),
    TransformStep("core.categories", """
        INSERT INTO core.categories (category_name, category_name_english)
        SELECT c.name, t.product_category_name_english
        FROM (
            SELECT product_category_name AS name FROM staging.category_translation
            UNION
            SELECT product_category_name FROM staging.products WHERE product_category_name IS NOT NULL
        ) c
        LEFT JOIN staging.category_translation t ON t.product_category_name = c.name
    """),
    TransformStep("core.products", """
        INSERT INTO core.products (product_id, category_name, name_length, description_length,
                                   photos_qty, weight_g, length_cm, height_cm, width_cm)
        SELECT product_id, NULLIF(product_category_name, ''),
               NULLIF(product_name_lenght, '')::numeric::int,
               NULLIF(product_description_lenght, '')::numeric::int,
               NULLIF(product_photos_qty, '')::numeric::int,
               NULLIF(product_weight_g, '')::numeric::int,
               NULLIF(product_length_cm, '')::numeric::int,
               NULLIF(product_height_cm, '')::numeric::int,
               NULLIF(product_width_cm, '')::numeric::int
        FROM staging.products
    """),
    TransformStep("core.orders", """
        INSERT INTO core.orders (order_id, customer_id, order_status, purchased_at, approved_at,
                                 delivered_carrier_at, delivered_customer_at, estimated_delivery_date)
        SELECT order_id, customer_id, order_status,
               order_purchase_timestamp::timestamp,
               NULLIF(order_approved_at, '')::timestamp,
               NULLIF(order_delivered_carrier_date, '')::timestamp,
               NULLIF(order_delivered_customer_date, '')::timestamp,
               order_estimated_delivery_date::date
        FROM staging.orders
    """),
    TransformStep("core.order_items", """
        INSERT INTO core.order_items (order_id, order_item_id, product_id, seller_id,
                                      shipping_limit_at, price, freight_value)
        SELECT order_id, order_item_id::smallint, product_id, seller_id,
               shipping_limit_date::timestamp, price::numeric(10,2), freight_value::numeric(10,2)
        FROM staging.order_items
    """),
    TransformStep("core.payments", """
        INSERT INTO core.payments (order_id, payment_sequential, payment_type, installments, amount)
        SELECT order_id, payment_sequential::smallint, payment_type,
               payment_installments::smallint, payment_value::numeric(10,2)
        FROM staging.order_payments
    """),
    TransformStep("core.reviews", """
        INSERT INTO core.reviews (review_id, score, title, message, created_at, answered_at)
        SELECT DISTINCT ON (review_id)
               review_id, review_score::smallint,
               NULLIF(review_comment_title, ''), NULLIF(review_comment_message, ''),
               review_creation_date::timestamp, NULLIF(review_answer_timestamp, '')::timestamp
        FROM staging.order_reviews
        ORDER BY review_id
    """),
    TransformStep("core.order_reviews", """
        INSERT INTO core.order_reviews (review_id, order_id)
        SELECT DISTINCT review_id, order_id FROM staging.order_reviews
    """),
]


class CoreLoader:

    def __init__(self, db: Database, steps: list[TransformStep] | None = None) -> None:
        self._db = db
        self._steps = steps if steps is not None else STEPS

    def run(self) -> dict[str, int]:
        """Execute every step and return the number of rows loaded per table."""
        self._prepare()
        counts: dict[str, int] = {}
        for step in self._steps:
            start = time.perf_counter()
            counts[step.target] = self._db.execute(step.sql)
            logger.info("%-22s %8d rows  %.2fs",
                        step.target, counts[step.target], time.perf_counter() - start)
        return counts

    def _prepare(self) -> None:
        self._db.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
        targets = ", ".join(step.target for step in self._steps)
        self._db.execute(f"TRUNCATE {targets}")