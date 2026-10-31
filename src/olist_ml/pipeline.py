"""Deployable predictor: preprocessing + model + decision threshold, saved as one artifact."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from olist_ml.features.preprocessing import Preprocessor
from olist_ml.models.base import BaseModel

# Features derived from raw order fields. They must match sql/06_features.sql exactly.
DERIVED_FEATURES = ["purchase_month", "purchase_weekday", "purchase_hour",
                    "estimated_days", "freight_ratio", "same_state"]


def features_from_orders(orders: list[dict]) -> pd.DataFrame:
    """Derive model features from raw order fields, with the same definitions as the SQL pipeline."""
    df = pd.DataFrame(orders)
    purchased = pd.to_datetime(df["purchased_at"])
    if purchased.dt.tz is not None:
        purchased = purchased.dt.tz_localize(None)
    df["purchase_month"] = purchased.dt.month
    df["purchase_weekday"] = purchased.dt.day_of_week + 1          # ISO weekday: Monday = 1 (isodow)
    df["purchase_hour"] = purchased.dt.hour
    df["estimated_days"] = (pd.to_datetime(df["estimated_delivery_date"]) - purchased.dt.normalize()).dt.days
    df["freight_ratio"] = df["total_freight"] / df["total_price"]
    df["same_state"] = df["customer_state"] == df["seller_state"]
    return df


@dataclass
class LatePredictor:
    preprocessor: Preprocessor
    model: BaseModel
    threshold: float
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, orders: list[dict]) -> np.ndarray:
        """Probability of late delivery for raw orders (dicts with the API's input fields)."""
        return self.model.predict_proba(self.preprocessor.transform(features_from_orders(orders)))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> LatePredictor:
        """Only load artifacts you created yourself: unpickling a file can execute code."""
        with Path(path).open("rb") as f:
            predictor = pickle.load(f)
        if not isinstance(predictor, cls):
            raise TypeError(f"{path} does not contain a LatePredictor")
        return predictor