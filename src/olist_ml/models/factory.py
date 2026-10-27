"""Factory: build any model by name, so scripts do not depend on concrete classes."""
from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.tree import DecisionTreeClassifier

from olist_ml.models.base import BaseModel
from olist_ml.models.decision_tree import DecisionTree
from olist_ml.models.logistic_regression import LogisticRegression
from olist_ml.models.neural_net import NeuralNetwork
from olist_ml.models.sklearn_adapter import SklearnModel


def _sk_logistic_regression(C: float = 1.0) -> BaseModel:
    return SklearnModel(SkLogisticRegression(C=C, max_iter=5000), "sk_logistic_regression")


def _sk_decision_tree(max_depth: int = 8, min_samples_leaf: int = 100) -> BaseModel:
    return SklearnModel(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                               random_state=0), "sk_decision_tree")


def _sk_gradient_boosting(max_iter: int = 300, learning_rate: float = 0.05) -> BaseModel:
    return SklearnModel(HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate,
                                                       early_stopping=False, random_state=0),
                        "sk_gradient_boosting")


MODEL_REGISTRY: dict[str, Callable[..., BaseModel]] = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTree,
    "neural_network": NeuralNetwork,
    "sk_logistic_regression": _sk_logistic_regression,
    "sk_decision_tree": _sk_decision_tree,
    "sk_gradient_boosting": _sk_gradient_boosting,
}


def create_model(name: str, **params) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)