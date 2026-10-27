"""Factory: build any model by name, so scripts do not depend on concrete classes."""
from __future__ import annotations

from olist_ml.models.base import BaseModel
from olist_ml.models.decision_tree import DecisionTree
from olist_ml.models.logistic_regression import LogisticRegression
from olist_ml.models.neural_net import NeuralNetwork

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTree,
    "neural_network": NeuralNetwork,
}


def create_model(name: str, **params) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)
