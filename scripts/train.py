"""Train the from-scratch logistic regression. Usage: python scripts/train.py"""
import logging
import time
from pathlib import Path

import numpy as np

from olist_ml.evaluation.metrics import summary
from olist_ml.features.dataset import TARGET, CsvFeatureSource, time_split
from olist_ml.features.preprocessing import Preprocessor
from olist_ml.models.logistic_regression import LogisticRegression
from olist_ml.viz.plots import plot_loss_curve

DATA = Path("data/processed/order_features.csv")
FIGURES = Path("docs/figures")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = CsvFeatureSource(DATA).load()
    train, val, test = time_split(df)
    for name, part in (("train", train), ("validation", val), ("test", test)):
        logging.info("%-10s %6d orders  %s to %s  late %.2f%%", name, len(part),
                     part["purchased_at"].min().date(), part["purchased_at"].max().date(),
                     100 * part[TARGET].mean())

    prep = Preprocessor()
    X_train, X_val, X_test = prep.fit_transform(train), prep.transform(val), prep.transform(test)
    y_train, y_val, y_test = (part[TARGET].to_numpy(dtype=int) for part in (train, val, test))
    logging.info("Features after preprocessing: %d", X_train.shape[1])

    model = LogisticRegression()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    logging.info("Trained in %.1f s, %d epochs, final loss %.4f",
                 time.perf_counter() - start, len(model.loss_history_), model.loss_history_[-1])

    for name, X, y in (("validation", X_val, y_val), ("test", X_test, y_test)):
        m = summary(y, model.predict_proba(X))
        logging.info("%-10s ROC-AUC %.3f | PR-AUC %.3f (random %.3f) | "
                     "precision %.3f recall %.3f F1 %.3f at threshold 0.5",
                     name, m["roc_auc"], m["pr_auc"], y.mean(), m["precision"], m["recall"], m["f1"])

    logging.info("\nLargest coefficients (positive = more likely late):")
    for i in np.argsort(-np.abs(model.w_))[:12]:
        logging.info("  %+.3f  %s", model.w_[i], prep.feature_names_[i])

    plot_loss_curve(model.loss_history_, FIGURES / "logreg_loss.png")


if __name__ == "__main__":
    main()