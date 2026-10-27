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
    """Area under the precision-recall step curve: AP = sum_n (R_n - R_(n-1)) * P_n.

    Computed over distinct thresholds, so tied scores are treated as one group and the
    result does not depend on the arbitrary order of tied rows.
    """
    tp, fp, _, n_pos, _ = _ranked_counts(y, scores)
    precision, recall = tp / (tp + fp), tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


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
    
def _ranked_counts(y: np.ndarray, scores: np.ndarray):
    """Cumulative TP and FP when predicting 'late' for every score >= each distinct threshold."""
    y = np.asarray(y).astype(bool)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s, hits = scores[order], y[order]
    last_of_group = np.r_[s[1:] != s[:-1], True]       # tied scores share one threshold
    tp = np.cumsum(hits)[last_of_group]
    fp = np.cumsum(~hits)[last_of_group]
    return tp, fp, s[last_of_group], hits.sum(), (~hits).sum()


def roc_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp, fp, _, n_pos, n_neg = _ranked_counts(y, scores)
    return np.r_[0.0, fp / n_neg], np.r_[0.0, tp / n_pos]


def pr_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp, fp, thresholds, n_pos, _ = _ranked_counts(y, scores)
    return tp / (tp + fp), tp / n_pos, thresholds


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold that maximizes F1 (choose it on validation data, never on test data)."""
    precision, recall, thresholds = pr_curve(y, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1)])