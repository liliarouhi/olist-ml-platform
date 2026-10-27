"""Model dataset (one row per delivered order) with interchangeable sources."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from olist_ml.config import DatabaseConfig
from olist_ml.db.connection import Database

TARGET = "is_late"
NUMERIC_FEATURES = [
    "estimated_days", "n_items", "n_sellers", "total_price", "total_freight", "freight_ratio",
    "total_weight_g", "total_volume_cm3", "max_photos", "installments", "distance_km",
    "seller_prior_deliveries", "seller_prior_late",
    "purchase_month", "purchase_weekday", "purchase_hour",
]
CATEGORICAL_FEATURES = ["main_category", "payment_type", "customer_state", "seller_state"]
BOOLEAN_FEATURES = ["same_state"]


class FeatureSource(ABC):
    """Contract for anything that can provide the model dataset."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return the dataset sorted by purchase time, with consistent column types."""


class PostgresFeatureSource(FeatureSource):
    """Reads ml.order_features from PostgreSQL."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self._config = config or DatabaseConfig()

    def load(self) -> pd.DataFrame:
        with Database(self._config) as db, db.connection.cursor() as cur:
            cur.execute("SELECT * FROM ml.order_features ORDER BY purchased_at")
            columns = [col.name for col in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=columns)
        return _with_types(df)


class CsvFeatureSource(FeatureSource):
    """Reads an exported copy, so models can be trained without a running database."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._path, parse_dates=["purchased_at"])
        return _with_types(df).sort_values("purchased_at", ignore_index=True)


def _with_types(df: pd.DataFrame) -> pd.DataFrame:
    """Same column types whatever the source: floats (NaN for missing), booleans, categories."""
    df = df.copy()
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    df[TARGET] = df[TARGET].astype(bool)
    df[BOOLEAN_FEATURES] = df[BOOLEAN_FEATURES].astype(bool)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df

def time_split(df: pd.DataFrame, validation_start: str = "2018-03-01",
               test_start: str = "2018-06-01") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split: the model never trains on orders newer than those it is evaluated on."""
    t = df["purchased_at"]
    return (df[t < validation_start],
            df[(t >= validation_start) & (t < test_start)],
            df[t >= test_start])