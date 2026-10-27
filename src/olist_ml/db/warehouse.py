"""Read-only data access for the dashboard (Repository pattern over the dw star schema)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

import pandas as pd

from olist_ml.config import DatabaseConfig
from olist_ml.db.connection import Database

# Delivered order items matching the dashboard filters, then aggregated to one row per order.
# Every query in WarehouseRepository starts from these two CTEs.
FILTERED = """
WITH items AS (
    SELECT f.order_id, f.price::float AS price, f.is_late, f.delivery_days,
           f.review_score::float AS review_score,
           d.month_start, d.day_of_week, c.state, p.category
    FROM dw.fact_order_items f
    JOIN dw.dim_date d     USING (date_key)
    JOIN dw.dim_customer c USING (customer_key)
    JOIN dw.dim_product p  USING (product_key)
    WHERE f.order_status = 'delivered'
      AND f.purchased_date BETWEEN %(start)s AND %(end)s
      AND (%(states)s::text[] IS NULL OR c.state = ANY(%(states)s::text[]))
      AND (%(categories)s::text[] IS NULL OR p.category = ANY(%(categories)s::text[]))
),
orders AS (
    SELECT order_id, min(state) AS state, sum(price) AS revenue, bool_or(is_late) AS is_late,
           max(delivery_days) AS delivery_days, max(review_score) AS review_score,
           min(day_of_week) AS day_of_week, min(month_start) AS month_start
    FROM items
    GROUP BY order_id
)
"""


@dataclass(frozen=True)
class Filters:
    """Dashboard filters. Empty tuples mean 'all'."""

    start: date
    end: date
    states: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()

    def params(self) -> dict:
        return {"start": self.start, "end": self.end,
                "states": list(self.states) or None,
                "categories": list(self.categories) or None}

    def previous_period(self) -> Filters:
        """The period of equal length right before this one, for KPI comparisons."""
        length = self.end - self.start
        end = self.start - timedelta(days=1)
        return replace(self, start=end - length, end=end)


class WarehouseRepository:
    """Read-only queries for the dashboard. Every method returns a pandas DataFrame."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self._config = config or DatabaseConfig()

    def _frame(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        with Database(self._config) as db, db.connection.cursor() as cur:
            cur.execute(sql, params)
            columns = [col.name for col in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=columns)

    def options(self) -> tuple[list[str], list[str]]:
        states = self._frame("SELECT DISTINCT state FROM dw.dim_customer ORDER BY 1")["state"]
        categories = self._frame(
            "SELECT category FROM dw.mv_monthly_category_revenue GROUP BY 1 ORDER BY sum(revenue) DESC"
        )["category"]
        return states.tolist(), categories.tolist()

    def kpis(self, f: Filters) -> pd.DataFrame:
        return self._frame(FILTERED + """
            SELECT coalesce(sum(revenue), 0)            AS revenue,
                   count(*)                             AS orders,
                   avg(revenue)                         AS avg_order_value,
                   (100 * avg(is_late::int))::float     AS late_pct,
                   avg(review_score)                    AS avg_review,
                   avg(delivery_days)::float            AS avg_delivery_days
            FROM orders""", f.params())

    def monthly(self, f: Filters) -> pd.DataFrame:
        return self._frame(FILTERED + """
            SELECT month_start, sum(revenue) AS revenue, count(*) AS orders,
                   (100 * avg(is_late::int))::float AS late_pct
            FROM orders GROUP BY 1 ORDER BY 1""", f.params())

    def weekday(self, f: Filters) -> pd.DataFrame:
        return self._frame(FILTERED + """
            SELECT day_of_week, count(*) AS orders, sum(revenue) AS revenue
            FROM orders GROUP BY 1 ORDER BY 1""", f.params())

    def states(self, f: Filters) -> pd.DataFrame:
        return self._frame(FILTERED + """
            SELECT state, count(*) AS orders, sum(revenue) AS revenue,
                   (100 * avg(is_late::int))::float AS late_pct,
                   avg(delivery_days)::float        AS avg_delivery_days,
                   avg(review_score)                AS avg_review
            FROM orders GROUP BY 1""", f.params())

    def review_by_delivery(self, f: Filters) -> pd.DataFrame:
        return self._frame(FILTERED + """
            SELECT least(delivery_days, 40) AS delivery_days, count(*) AS orders,
                   avg(review_score) AS avg_review
            FROM orders
            WHERE delivery_days IS NOT NULL AND review_score IS NOT NULL
            GROUP BY 1 ORDER BY 1""", f.params())

    def review_distribution(self, f: Filters) -> pd.DataFrame:
        return self._frame(FILTERED + """
            SELECT CASE WHEN is_late THEN 'Late' ELSE 'On time' END AS delivery,
                   round(review_score)::int AS score, count(*) AS orders
            FROM orders
            WHERE is_late IS NOT NULL AND review_score IS NOT NULL
            GROUP BY 1, 2 ORDER BY 1, 2""", f.params())

    def top_categories(self, f: Filters, limit: int = 15) -> pd.DataFrame:
        return self._frame(FILTERED + """
            SELECT category, sum(price) AS revenue, count(DISTINCT order_id) AS orders
            FROM items GROUP BY 1 ORDER BY 2 DESC LIMIT %(limit)s""", {**f.params(), "limit": limit})

    def category_month(self, f: Filters, limit: int = 8) -> pd.DataFrame:
        return self._frame(FILTERED + """,
            top AS (SELECT category FROM items GROUP BY 1 ORDER BY sum(price) DESC LIMIT %(limit)s)
            SELECT i.month_start, i.category, sum(i.price) AS revenue
            FROM items i JOIN top USING (category)
            GROUP BY 1, 2""", {**f.params(), "limit": limit})