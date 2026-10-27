"""Evaluation metrics implemented from scratch."""
from __future__ import annotations

import numpy as np
import pandas as pd


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), via the Mann-Whitney U statistic."""
    y = np.asarray(y).astype(bool)
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank().to_numpy()  # ties get average ranks
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve: mean precision at the rank of each true positive."""
    y = np.asarray(y).astype(bool)
    hits = y[np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")]
    precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision_at_k[hits].sum() / hits.sum())


def precision_recall_f1(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y, p = np.asarray(y).astype(bool), np.asarray(y_pred).astype(bool)
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def summary(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    precision, recall, f1 = precision_recall_f1(y, np.asarray(scores) >= threshold)
    return {"roc_auc": roc_auc(y, scores), "pr_auc": average_precision(y, scores),
            "precision": precision, "recall": recall, "f1": f1}