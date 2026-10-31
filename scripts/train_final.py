"""Train the deployable predictor, verify training-serving consistency, and save it.

Usage: python scripts/train_final.py [model_name]   (default: sk_gradient_boosting)
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from olist_ml.evaluation.metrics import average_precision, best_f1_threshold, precision_recall_f1, roc_auc
from olist_ml.features.dataset import TARGET, CsvFeatureSource, time_split
from olist_ml.features.preprocessing import Preprocessor
from olist_ml.models.factory import create_model
from olist_ml.pipeline import DERIVED_FEATURES, LatePredictor

DATA = Path("data/processed/order_features.csv")
ARTIFACT = Path("artifacts/late_predictor.pkl")
DEFAULT_MODEL = "sk_gradient_boosting"


def to_raw_orders(df: pd.DataFrame) -> list[dict]:
    """Rebuild the raw request fields an API client would send for these orders."""
    raw = df.drop(columns=DERIVED_FEATURES + [TARGET, "order_id"]).copy()
    raw["estimated_delivery_date"] = (df["purchased_at"].dt.normalize()
                                      + pd.to_timedelta(df["estimated_days"], unit="D")).dt.date
    return raw.to_dict("records")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    df = CsvFeatureSource(DATA).load()
    train, val, test = time_split(df)
    y_train, y_val, y_test = (part[TARGET].to_numpy(dtype=int) for part in (train, val, test))

    prep = Preprocessor()
    X_train = prep.fit_transform(train)
    model = create_model(name).fit(X_train, y_train, prep.transform(val), y_val)
    threshold = best_f1_threshold(y_val, model.predict_proba(prep.transform(val)))
    test_scores = model.predict_proba(prep.transform(test))
    precision, recall, f1 = precision_recall_f1(y_test, test_scores >= threshold)

    metadata = {
        "model": name,
        "trained_on": f"{train['purchased_at'].min().date()} to {train['purchased_at'].max().date()}",
        "threshold": round(threshold, 4),
        "n_features": X_train.shape[1],
        "test": {"pr_auc": round(average_precision(y_test, test_scores), 4),
                 "roc_auc": round(roc_auc(y_test, test_scores), 4),
                 "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                 "base_rate": round(float(y_test.mean()), 4)},
    }
    predictor = LatePredictor(prep, model, threshold, metadata)

    # Training-serving consistency: raw fields -> API feature code must give the same scores
    # as the features computed in SQL.
    sample = 2000
    api_scores = predictor.predict_proba(to_raw_orders(test.head(sample)))
    if not np.allclose(api_scores, test_scores[:sample]):
        raise SystemExit("Training-serving skew: API features differ from the SQL features")
    logging.info("Consistency check passed: API feature code reproduces the SQL features on %d orders", sample)

    predictor.save(ARTIFACT)
    logging.info("Saved %s\n%s", ARTIFACT, json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()