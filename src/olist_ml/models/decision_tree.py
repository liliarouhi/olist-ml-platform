"""CART-style decision tree for binary classification, implemented with NumPy only.

Splits minimize weighted Gini impurity; all thresholds of a feature are scored at once
with cumulative sums after sorting (O(n log n) per feature and node).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from olist_ml.models.base import BaseModel, balanced_weights


@dataclass
class Node:
    value: float                       # weighted share of positives (late) in this node
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def gini(positive, total):
    """Gini impurity 2q(1 - q) where q = positive / total (works on scalars and arrays)."""
    q = np.asarray(positive, dtype=float) / np.maximum(total, 1e-12)
    return 2.0 * q * (1.0 - q)


class DecisionTree(BaseModel):
    name = "decision_tree_scratch"

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 100,
                 min_gain: float = 1e-6, balanced: bool = True) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.balanced = balanced

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> DecisionTree:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        weights = balanced_weights(y) if self.balanced else np.ones(len(y))
        self.importances_ = np.zeros(X.shape[1])
        self.n_leaves_ = 0
        self.root_ = self._grow(X, y, weights, depth=0)
        if self.importances_.sum() > 0:
            self.importances_ /= self.importances_.sum()
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int) -> Node:
        total, positive = w.sum(), (w * y).sum()
        node = Node(value=float(positive / total))
        pure = positive == 0 or positive == total
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or pure:
            self.n_leaves_ += 1
            return node
        split = self._best_split(X, y, w)
        if split is None or split[2] < self.min_gain:
            self.n_leaves_ += 1
            return node
        feature, threshold, gain = split
        self.importances_[feature] += gain * total     # weighted impurity decrease
        mask = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(X[mask], y[mask], w[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], w[~mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        n = len(y)
        total, positive = w.sum(), (w * y).sum()
        left_sizes = np.arange(1, n)
        large_enough = (left_sizes >= self.min_samples_leaf) & (n - left_sizes >= self.min_samples_leaf)
        wy = w * y
        best, best_impurity = None, np.inf
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="mergesort")
            xs = X[order, j]
            left_w = np.cumsum(w[order])[:-1]            # weight left of each possible cut
            left_p = np.cumsum(wy[order])[:-1]
            right_w, right_p = total - left_w, positive - left_p
            impurity = (left_w * gini(left_p, left_w) + right_w * gini(right_p, right_w)) / total
            impurity[~(large_enough & (xs[:-1] < xs[1:]))] = np.inf   # only cuts between distinct values
            i = int(np.argmin(impurity))
            if impurity[i] < best_impurity:
                best_impurity, best = impurity[i], (j, (xs[i] + xs[i + 1]) / 2)
        if best is None:
            return None
        return best[0], best[1], float(gini(positive, total) - best_impurity)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X))
        self._route(self.root_, X, np.arange(len(X)), out)
        return out

    def _route(self, node: Node, X: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
        """Send all rows down the tree at once instead of looping row by row."""
        if node.is_leaf:
            out[idx] = node.value
            return
        go_left = X[idx, node.feature] <= node.threshold
        self._route(node.left, X, idx[go_left], out)
        self._route(node.right, X, idx[~go_left], out)