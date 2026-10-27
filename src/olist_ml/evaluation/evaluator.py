"""Model comparison with paired bootstrap confidence intervals."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from olist_ml.evaluation.metrics import average_precision


class Evaluator:
    """Resamples the evaluation set with replacement to measure how stable each metric is."""

    def __init__(self, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> None:
        self.n_boot = n_boot
        self.seed = seed
        self.alpha = alpha

    def bootstrap(self, y: np.ndarray, scores: dict[str, np.ndarray],
                  metric: Callable = average_precision) -> dict[str, np.ndarray]:
        """Paired bootstrap: each resample is applied to every model, so differences are comparable."""
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y)
        samples = {name: np.empty(self.n_boot) for name in scores}
        for b in range(self.n_boot):
            idx = rng.integers(0, len(y), len(y))
            for name, s in scores.items():
                samples[name][b] = metric(y[idx], s[idx])
        return samples

    def interval(self, samples: np.ndarray) -> tuple[float, float]:
        low, high = np.percentile(samples, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])
        return float(low), float(high)