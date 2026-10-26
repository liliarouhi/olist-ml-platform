"""Figures from the data warehouse materialized views. Usage: python scripts/plot_olap.py"""
import logging
from pathlib import Path

from olist_ml.db.connection import Database
from olist_ml.viz.plots import plot_category_heatmap, plot_state_delivery

OUT_DIR = Path("docs/figures")
TOP_N = 10


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with Database() as db:
        conn = db.connection
        top = [row[0] for row in conn.execute(
            "SELECT category FROM dw.mv_monthly_category_revenue "
            "GROUP BY 1 ORDER BY sum(revenue) DESC LIMIT %s", (TOP_N,)).fetchall()]
        months = [row[0] for row in conn.execute(
            "SELECT DISTINCT month_start FROM dw.mv_monthly_category_revenue ORDER BY 1").fetchall()]
        revenue = {(m, c): float(v) for m, c, v in conn.execute(
            "SELECT month_start, category, revenue FROM dw.mv_monthly_category_revenue "
            "WHERE category = ANY(%s)", (top,)).fetchall()}
        states = conn.execute(
            "SELECT state, delivered_orders, late_pct, avg_delivery_days "
            "FROM dw.mv_state_delivery ORDER BY state").fetchall()

    matrix = [[revenue.get((m, c), 0.0) for m in months] for c in top]
    plot_category_heatmap(months, top, matrix, OUT_DIR / "category_revenue_heatmap.png")
    plot_state_delivery([s[0] for s in states], [s[1] for s in states],
                        [float(s[2]) for s in states], [float(s[3]) for s in states],
                        OUT_DIR / "state_delivery_scatter.png")
    logging.info("Saved figures to %s", OUT_DIR)


if __name__ == "__main__":
    main()