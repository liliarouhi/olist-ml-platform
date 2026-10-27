"""Train and compare the from-scratch models. Usage: python scripts/train.py"""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from olist_ml.evaluation.metrics import summary
from olist_ml.features.dataset import TARGET, CsvFeatureSource, time_split
from olist_ml.features.preprocessing import Preprocessor
from olist_ml.models.factory import create_model
from olist_ml.viz.plots import plot_feature_importance, plot_loss_curve, plot_nn_training

DATA = Path("data/processed/order_features.csv")
FIGURES = Path("docs/figures")
MODELS = ["logistic_regression", "decision_tree", "neural_network"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = CsvFeatureSource(DATA).load()
    train, val, test = time_split(df)
    prep = Preprocessor()
    X_train, X_val, X_test = prep.fit_transform(train), prep.transform(val), prep.transform(test)
    y_train, y_val, y_test = (part[TARGET].to_numpy(dtype=int) for part in (train, val, test))
    logging.info("Train %d | validation %d | test %d orders, %d features",
                 len(y_train), len(y_val), len(y_test), X_train.shape[1])

    rows = []
    for name in MODELS:
        model = create_model(name)
        start = time.perf_counter()
        model.fit(X_train, y_train, X_val, y_val)
        seconds = time.perf_counter() - start
        logging.info("\n%s trained in %.1f s", name, seconds)
        for split, X, y in (("validation", X_val, y_val), ("test", X_test, y_test)):
            m = summary(y, model.predict_proba(X))
            rows.append({"model": name, "set": split, "seconds": round(seconds, 1),
                         **{k: round(v, 4) for k, v in m.items()}})
            logging.info("  %-10s ROC-AUC %.3f | PR-AUC %.3f (random %.3f)",
                         split, m["roc_auc"], m["pr_auc"], y.mean())

        if name == "logistic_regression":
            plot_loss_curve(model.loss_history_, FIGURES / "logreg_loss.png")
        elif name == "decision_tree":
            logging.info("  leaves: %d", model.n_leaves_)
            for i in np.argsort(-model.importances_)[:8]:
                logging.info("  importance %.3f  %s", model.importances_[i], prep.feature_names_[i])
            plot_feature_importance(prep.feature_names_, model.importances_, FIGURES / "tree_importance.png")
        elif name == "neural_network":
            logging.info("  epochs run: %d, best validation PR-AUC %.3f",
                         len(model.loss_history_), max(model.val_history_))
            plot_nn_training(model.loss_history_, model.val_history_, FIGURES / "nn_training.png")

    table = pd.DataFrame(rows)
    table.to_csv(FIGURES / "model_comparison.csv", index=False)
    logging.info("\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()