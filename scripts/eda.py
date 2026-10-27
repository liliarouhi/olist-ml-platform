"""Exploratory analysis of the model dataset. Usage: python scripts/eda.py

Outputs: data/processed/order_features.csv (not in git),
         docs/figures/eda_late_rate.png, docs/figures/eda_correlation.png
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from olist_ml.features.dataset import NUMERIC_FEATURES, TARGET, PostgresFeatureSource
from olist_ml.viz.plots import plot_correlation, plot_late_rate_panels

logger = logging.getLogger(__name__)
FIGURES = Path("docs/figures")
PROCESSED = Path("data/processed")


def late_rate_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Late rate (%) and number of orders per group for the most promising features."""
    seller_rate = df["seller_prior_late"] / df["seller_prior_deliveries"].where(df["seller_prior_deliveries"] > 0)
    seller_bucket = pd.cut(seller_rate, [-0.001, 0, 0.05, 0.10, 0.20, 1.0],
                           labels=["0%", "0-5%", "5-10%", "10-20%", ">20%"])
    seller_bucket = seller_bucket.cat.add_categories("no history").fillna("no history")
    groups = {
        "Promised delivery time (days)": pd.cut(df["estimated_days"], [0, 10, 15, 20, 25, 30, 40, 200]),
        "Customer-seller distance (km)": pd.cut(df["distance_km"], [0, 100, 250, 500, 1000, 2000, 5000]),
        "Purchase month": df["purchase_month"].astype(int),
        "Seller's late rate before the order": seller_bucket,
    }
    return {title: df.groupby(key, observed=True)[TARGET].agg(late_pct=lambda s: 100 * s.mean(),
                                                               orders="size")
            for title, key in groups.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = PostgresFeatureSource().load()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "order_features.csv", index=False)

    base_rate = 100 * df[TARGET].mean()
    logger.info("Orders: %d | late: %d (%.2f%%)", len(df), df[TARGET].sum(), base_rate)
    missing = df.isna().sum()
    logger.info("Missing values:\n%s", missing[missing > 0].to_string())

    tables = late_rate_tables(df)
    for title, table in tables.items():
        logger.info("\n%s\n%s", title, table.round(2).to_string())

    corr = df[NUMERIC_FEATURES + [TARGET]].astype(float).corr(method="spearman")[TARGET].drop(TARGET)
    logger.info("\nSpearman correlation with %s:\n%s", TARGET, corr.sort_values().round(3).to_string())

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_late_rate_panels(tables, base_rate, FIGURES / "eda_late_rate.png")
    plot_correlation(df, NUMERIC_FEATURES + [TARGET], FIGURES / "eda_correlation.png")
    logger.info("\nSaved figures to %s", FIGURES)


if __name__ == "__main__":
    main()