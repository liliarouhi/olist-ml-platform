"""Logistic regression trained with batch gradient descent, implemented with NumPy only.

Loss: weighted binary cross-entropy + (l2 / 2) * ||w||^2.  Derivation: docs/08_ml_from_scratch.md
"""
from __future__ import annotations

import numpy as np

from olist_ml.models.base import BaseModel, balanced_weights


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function: 1 / (1 + e^-z) = (1 + tanh(z / 2)) / 2."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class LogisticRegression(BaseModel):
    name = "logistic_regression_scratch"

    def __init__(self, learning_rate: float = 0.5, max_epochs: int = 3000, l2: float = 1e-3,
                 tol: float = 1e-7, balanced: bool = True) -> None:
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.l2 = l2
        self.tol = tol
        self.balanced = balanced
        self.loss_history_: list[float] = []

    def loss_and_gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                          sample_weight: np.ndarray | None = None) -> tuple[float, np.ndarray, float]:
        """Loss and its exact gradient with respect to w and b."""
        s = np.ones(len(y)) if sample_weight is None else sample_weight
        total = s.sum()
        z = X @ w + b
        per_row = y * np.logaddexp(0.0, -z) + (1 - y) * np.logaddexp(0.0, z)  # stable -log-likelihood
        loss = (s @ per_row) / total + 0.5 * self.l2 * (w @ w)
        residual = s * (sigmoid(z) - y) / total          # dL/dz for each row
        return loss, X.T @ residual + self.l2 * w, residual.sum()

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> LogisticRegression:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        weights = balanced_weights(y) if self.balanced else np.ones(len(y))
        self.w_, self.b_ = np.zeros(X.shape[1]), 0.0
        self.loss_history_ = []
        previous = np.inf
        for _ in range(self.max_epochs):
            loss, grad_w, grad_b = self.loss_and_gradient(X, y, self.w_, self.b_, weights)
            self.loss_history_.append(float(loss))
            if previous - loss < self.tol:   # converged (or diverging): stop
                break
            previous = loss
            self.w_ -= self.learning_rate * grad_w
            self.b_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(X, dtype=float) @ self.w_ + self.b_)