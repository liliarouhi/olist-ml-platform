"""Tests for the from-scratch model and metrics. Run: pytest -q"""
import numpy as np

from olist_ml.evaluation.metrics import average_precision, roc_auc
from olist_ml.models.logistic_regression import LogisticRegression, sigmoid
from olist_ml.models.decision_tree import DecisionTree
from olist_ml.models.neural_net import NeuralNetwork

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
    
def test_neural_network_gradient_check():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(30, 3)), (rng.random(30) < 0.4).astype(float)
    s = rng.random(30) + 0.5
    net = NeuralNetwork(hidden=(4,), l2=0.01)
    params = net._init_params(3, rng)
    _, grads = net.loss_and_gradients(X, y, params, s)
    eps = 1e-6
    for p, g in zip(params, grads):
        flat, flat_grad = p.reshape(-1), g.reshape(-1)   # views: editing flat edits p
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + eps
            up = net.loss_and_gradients(X, y, params, s)[0]
            flat[i] = original - eps
            down = net.loss_and_gradients(X, y, params, s)[0]
            flat[i] = original
            assert abs((up - down) / (2 * eps) - flat_grad[i]) < 1e-5


def test_neural_network_learns_xor_but_logistic_regression_cannot():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, size=(600, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)          # not linearly separable
    net = NeuralNetwork(hidden=(16,), epochs=200, batch_size=64, learning_rate=0.01, patience=1000).fit(X, y)
    linear = LogisticRegression(max_epochs=2000).fit(X, y)
    assert (net.predict(X) == y).mean() > 0.9
    assert (linear.predict(X) == y).mean() < 0.7


def test_decision_tree_finds_the_true_threshold():
    X = np.arange(200, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 120).astype(int)
    tree = DecisionTree(max_depth=2, min_samples_leaf=5).fit(X, y)
    assert tree.root_.threshold == 119.5
    assert (tree.predict(X) == y).all()