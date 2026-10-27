"""Adapter pattern: use any scikit-learn classifier through the project's BaseModel interface."""
from __future__ import annotations

import numpy as np

from olist_ml.models.base import BaseModel, balanced_weights


class SklearnModel(BaseModel):
    """Wraps a scikit-learn estimator; trained with the same balanced weights as our models."""

    def __init__(self, estimator, name: str, balanced: bool = True) -> None:
        self.estimator = estimator
        self.name = name
        self.balanced = balanced

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> SklearnModel:
        y = np.asarray(y).astype(int)
        weights = balanced_weights(y) if self.balanced else None
        self.estimator.fit(X, y, sample_weight=weights)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]