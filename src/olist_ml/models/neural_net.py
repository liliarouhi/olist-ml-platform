"""Feed-forward neural network (ReLU hidden layers, sigmoid output) with backpropagation and Adam.

Implemented with NumPy only. Derivation: docs/08_ml_from_scratch.md
"""
from __future__ import annotations

import numpy as np

from olist_ml.evaluation.metrics import average_precision
from olist_ml.models.base import BaseModel, balanced_weights
from olist_ml.models.logistic_regression import sigmoid


class Adam:
    """Adam optimizer: per-parameter step sizes from running averages of g and g²."""

    def __init__(self, params: list[np.ndarray], learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.learning_rate, self.beta1, self.beta2, self.eps = learning_rate, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)    # bias correction
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)   # in place


class NeuralNetwork(BaseModel):
    name = "neural_network_scratch"

    def __init__(self, hidden: tuple[int, ...] = (32, 16), learning_rate: float = 1e-3,
                 l2: float = 1e-4, epochs: int = 40, batch_size: int = 512, patience: int = 5,
                 balanced: bool = True, seed: int = 0) -> None:
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.balanced = balanced
        self.seed = seed

    def _init_params(self, n_inputs: int, rng: np.random.Generator) -> list[np.ndarray]:
        """He initialization: W ~ N(0, 2 / fan_in), biases zero. Order: W1, b1, W2, b2, ..."""
        sizes = [n_inputs, *self.hidden, 1]
        params: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            params += [rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)), np.zeros(fan_out)]
        return params

    @staticmethod
    def _forward(X: np.ndarray, params: list[np.ndarray]):
        """Return pre-activations z_k and activations a_k (a_0 = X, last = output logits)."""
        pre, acts = [], [X]
        n_layers = len(params) // 2
        for k in range(n_layers):
            z = acts[-1] @ params[2 * k] + params[2 * k + 1]
            pre.append(z)
            acts.append(np.maximum(z, 0.0) if k < n_layers - 1 else z)
        return pre, acts

    def loss_and_gradients(self, X: np.ndarray, y: np.ndarray, params: list[np.ndarray],
                           sample_weight: np.ndarray | None = None):
        s = np.ones(len(y)) if sample_weight is None else sample_weight
        total = s.sum()
        pre, acts = self._forward(X, params)
        z = acts[-1][:, 0]
        per_row = y * np.logaddexp(0.0, -z) + (1 - y) * np.logaddexp(0.0, z)
        loss = (s @ per_row) / total + 0.5 * self.l2 * sum((W * W).sum() for W in params[0::2])

        delta = (s * (sigmoid(z) - y) / total)[:, None]   # dL/dz of the output layer
        grads: list[np.ndarray] = [np.empty(0)] * len(params)
        for k in reversed(range(len(params) // 2)):
            W = params[2 * k]
            grads[2 * k] = acts[k].T @ delta + self.l2 * W
            grads[2 * k + 1] = delta.sum(axis=0)
            if k > 0:
                delta = (delta @ W.T) * (pre[k - 1] > 0)   # chain rule through ReLU
        return float(loss), grads

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> NeuralNetwork:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        self.params_ = self._init_params(X.shape[1], rng)
        weights = balanced_weights(y) if self.balanced else np.ones(len(y))
        optimizer = Adam(self.params_, self.learning_rate)
        self.loss_history_: list[float] = []
        self.val_history_: list[float] = []
        best_score, best_params, waited = -np.inf, None, 0

        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            epoch_loss = 0.0
            for start in range(0, len(y), self.batch_size):
                batch = order[start:start + self.batch_size]
                loss, grads = self.loss_and_gradients(X[batch], y[batch], self.params_, weights[batch])
                optimizer.step(self.params_, grads)
                epoch_loss += loss * len(batch)
            self.loss_history_.append(epoch_loss / len(y))

            if X_val is not None:   # early stopping on validation PR-AUC
                score = average_precision(y_val, self.predict_proba(X_val))
                self.val_history_.append(score)
                if score > best_score:
                    best_score, best_params, waited = score, [p.copy() for p in self.params_], 0
                else:
                    waited += 1
                    if waited >= self.patience:
                        break
        if best_params is not None:
            self.params_ = best_params
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, acts = self._forward(np.asarray(X, dtype=float), self.params_)
        return sigmoid(acts[-1][:, 0])