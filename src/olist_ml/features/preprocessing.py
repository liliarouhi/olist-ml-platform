"""Feature preprocessing fitted on the training set only, so nothing leaks from validation/test."""
from __future__ import annotations

import numpy as np
import pandas as pd

from olist_ml.features.dataset import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

LOG_FEATURES = ["total_price", "total_freight", "total_weight_g", "total_volume_cm3",
                "seller_prior_deliveries"]
# Non-monotonic effect found in the EDA (docs/07): give linear models one indicator per range.
BINNED_FEATURES = {"estimated_days": [0, 10, 15, 20, 25, 30, 40, np.inf]}


class Preprocessor:
    """Turns the order DataFrame into a numeric matrix.

    fit() learns medians, means, standard deviations and category lists from the training data;
    transform() applies exactly those learned values to any other data.
    """

    def __init__(self, top_categories: int = 20) -> None:
        self.top_categories = top_categories
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        numeric = self._numeric(df)
        self.medians_ = numeric.median()
        self.missing_columns_ = [c for c in numeric.columns if numeric[c].isna().any()]
        filled = numeric.fillna(self.medians_)
        self.means_ = filled.mean()
        self.stds_ = filled.std().replace(0, 1.0)
        self.categories_ = {c: df[c].astype(str).value_counts().index[: self.top_categories].tolist()
                            for c in CATEGORICAL_FEATURES}
        self._fitted = True
        self.feature_names_ = list(self._build(df).columns)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() on the training data first.")
        return self._build(df).to_numpy(dtype=float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @staticmethod
    def _numeric(df: pd.DataFrame) -> pd.DataFrame:
        numeric = df[NUMERIC_FEATURES].astype(float).copy()
        for col in LOG_FEATURES:
            numeric[col] = np.log1p(numeric[col].clip(lower=0))  # compress skewed amounts
        return numeric

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = self._numeric(df)
        parts = [(numeric.fillna(self.medians_) - self.means_) / self.stds_,
                 pd.DataFrame({f"{c}_missing": numeric[c].isna().astype(float)
                               for c in self.missing_columns_}, index=df.index)]
        for col, edges in BINNED_FEATURES.items():
            parts.append(pd.get_dummies(pd.cut(df[col], edges), prefix=f"{col}_bin", dtype=float))
        for col in CATEGORICAL_FEATURES:
            text = df[col].astype(str)
            values = text.where(text.isin(self.categories_[col]), "other")  # rare and unseen -> other
            expected = [f"{col}_{v}" for v in self.categories_[col] + ["other"]]
            parts.append(pd.get_dummies(values, prefix=col, dtype=float)
                         .reindex(columns=expected, fill_value=0.0))
        parts.append(df[BOOLEAN_FEATURES].astype(float))
        return pd.concat(parts, axis=1)