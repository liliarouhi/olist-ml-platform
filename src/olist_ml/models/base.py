"""Common interface for every classifier in the project (from scratch or from a library)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Every model learns from (X, y) and returns the probability of the positive class (late)."""

    name: str = "model"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> BaseModel:
        """Learn parameters from training data. Validation data is optional (early stopping)."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability that each row belongs to the positive class."""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Sample weights n / (2 * n_class): both classes contribute equally to the loss."""
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    return (len(y) / (2 * counts))[y]