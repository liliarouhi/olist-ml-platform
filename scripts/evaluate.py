"""Evaluate all models: bootstrap confidence intervals, curves, threshold, error analysis, ablation.

Usage: python scripts/evaluate.py
"""
import logging
import time
from pathlib import Path

import pandas as pd

from olist_ml.evaluation.evaluator import Evaluator
from olist_ml.evaluation.metrics import (average_precision, best_f1_threshold, pr_curve,
                                         precision_recall_f1, roc_auc, roc_curve)
from olist_ml.features.dataset import TARGET, CsvFeatureSource, time_split
from olist_ml.features.preprocessing import Preprocessor
from olist_ml.models.factory import create_model
from olist_ml.viz.plots import plot_bootstrap_intervals, plot_curves

DATA = Path("data/processed/order_features.csv")
FIGURES = Path("docs/figures")
MODELS = ["logistic_regression", "decision_tree", "neural_network",
          "sk_logistic_regression", "sk_decision_tree", "sk_gradient_boosting"]
ABLATION_MODELS = ["decision_tree", "neural_network", "sk_gradient_boosting"]


def prepare(train, val, test, exclude=()):
    prep = Preprocessor(exclude=exclude)
    return prep.fit_transform(train), prep.transform(val), prep.transform(test)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = CsvFeatureSource(DATA).load()
    train, val, test = time_split(df)
    y_train, y_val, y_test = (part[TARGET].to_numpy(dtype=int) for part in (train, val, test))
    X_train, X_val, X_test = prepare(train, val, test)
    # Same objective as the scratch model: C = 1 / (lambda * sum of sample weights)
    params = {"sk_logistic_regression": {"C": 1 / (1e-3 * len(y_train))}}

    val_scores, test_scores = {}, {}
    for name in MODELS:
        start = time.perf_counter()
        model = create_model(name, **params.get(name, {})).fit(X_train, y_train, X_val, y_val)
        val_scores[name], test_scores[name] = model.predict_proba(X_val), model.predict_proba(X_test)
        logging.info("%-24s trained in %5.1f s", name, time.perf_counter() - start)

    # 1) Rank on validation; report test PR-AUC with paired bootstrap confidence intervals
    evaluator = Evaluator()
    samples = evaluator.bootstrap(y_test, test_scores)
    rows = []
    for name in MODELS:
        low, high = evaluator.interval(samples[name])
        rows.append({"model": name,
                     "val_pr_auc": average_precision(y_val, val_scores[name]),
                     "test_pr_auc": average_precision(y_test, test_scores[name]),
                     "test_pr_auc_low": low, "test_pr_auc_high": high,
                     "test_roc_auc": roc_auc(y_test, test_scores[name])})
    table = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False, ignore_index=True)
    table.round(4).to_csv(FIGURES / "evaluation_results.csv", index=False)
    logging.info("\nRanked by validation PR-AUC (test PR-AUC with 95%% bootstrap CI):\n%s",
                 table.round(3).to_string(index=False))

    best, runner_up = table["model"].iloc[0], table["model"].iloc[1]
    diff = samples[best] - samples[runner_up]
    low, high = evaluator.interval(diff)
    logging.info("\nSelected on validation: %s. Test PR-AUC difference vs %s: %+.3f, "
                 "95%% CI [%+.3f, %+.3f], better in %.0f%% of resamples",
                 best, runner_up, table["test_pr_auc"].iloc[0] - table["test_pr_auc"].iloc[1],
                 low, high, 100 * (diff > 0).mean())

    # 2) Threshold chosen on validation, applied once to test
    threshold = best_f1_threshold(y_val, val_scores[best])
    flagged = test_scores[best] >= threshold
    late = y_test == 1
    precision, recall, f1 = precision_recall_f1(y_test, flagged)
    logging.info("\nThreshold %.3f (max F1 on validation), applied to test: "
                 "precision %.3f, recall %.3f, F1 %.3f", threshold, precision, recall, f1)
    logging.info("Confusion matrix (test): TP %d | FP %d | FN %d | TN %d",
                 (flagged & late).sum(), (flagged & ~late).sum(),
                 (~flagged & late).sum(), (~flagged & ~late).sum())
    logging.info("Flags %.1f%% of orders and catches %d of %d late orders",
                 100 * flagged.mean(), (flagged & late).sum(), late.sum())

    # 3) Error analysis: where are late orders missed?
    late_orders = test.assign(flagged=flagged)[late]
    by_state = (late_orders.groupby("customer_state", observed=True)["flagged"]
                .agg(recall="mean", late_orders="size")
                .sort_values("late_orders", ascending=False).head(8))
    by_distance = (late_orders.groupby(pd.cut(late_orders["distance_km"], [0, 250, 500, 1000, 2000, 5000]),
                                       observed=True)["flagged"]
                   .agg(recall="mean", late_orders="size"))
    logging.info("\nRecall by customer state (8 states with most late orders):\n%s",
                 by_state.round(3).to_string())
    logging.info("\nRecall by distance (km):\n%s", by_distance.round(3).to_string())

    # 4) Ablation: retrain without purchase_month
    Xa_train, Xa_val, Xa_test = prepare(train, val, test, exclude=("purchase_month",))
    rows = []
    for name in ABLATION_MODELS:
        model = create_model(name).fit(Xa_train, y_train, Xa_val, y_val)
        rows.append({"model": name,
                     "val_with": average_precision(y_val, val_scores[name]),
                     "val_without": average_precision(y_val, model.predict_proba(Xa_val)),
                     "test_with": average_precision(y_test, test_scores[name]),
                     "test_without": average_precision(y_test, model.predict_proba(Xa_test))})
    logging.info("\nAblation, PR-AUC with and without purchase_month:\n%s",
                 pd.DataFrame(rows).round(3).to_string(index=False))

    # 5) Figures
    plot_curves({n: (roc_curve(y_test, s), pr_curve(y_test, s)) for n, s in test_scores.items()},
                y_test.mean(), FIGURES / "roc_pr_curves.png")
    plot_bootstrap_intervals(table, y_test.mean(), FIGURES / "pr_auc_intervals.png")


if __name__ == "__main__":
    main()