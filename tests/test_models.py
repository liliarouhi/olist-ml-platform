"""Tests for the from-scratch model and metrics. Run: pytest -q"""
import numpy as np

from olist_ml.evaluation.metrics import average_precision, roc_auc
from olist_ml.models.logistic_regression import LogisticRegression, sigmoid


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(50, 4)), (rng.random(50) < 0.3).astype(float)
    w, b, s = rng.normal(size=4), 0.3, rng.random(50) + 0.5
    model = LogisticRegression(l2=0.1)
    _, grad_w, grad_b = model.loss_and_gradient(X, y, w, b, s)

    def loss(w_, b_):
        return model.loss_and_gradient(X, y, w_, b_, s)[0]

    eps = 1e-6
    numeric_w = np.array([(loss(w + eps * e, b) - loss(w - eps * e, b)) / (2 * eps) for e in np.eye(4)])
    numeric_b = (loss(w, b + eps) - loss(w, b - eps)) / (2 * eps)
    assert np.allclose(grad_w, numeric_w, atol=1e-6)
    assert abs(grad_b - numeric_b) < 1e-6


def test_learns_linearly_separable_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(max_epochs=2000).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95


def test_sigmoid_is_numerically_stable():
    assert np.all(np.isfinite(sigmoid(np.array([-1000.0, 0.0, 1000.0]))))


def test_metrics_on_hand_computed_example():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert roc_auc(y, scores) == 0.75                                  # 3 of 4 pairs ordered correctly
    assert np.isclose(average_precision(y, scores), (1 / 1 + 2 / 3) / 2)