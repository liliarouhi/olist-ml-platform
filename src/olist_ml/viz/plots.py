"""Plotting helpers. Figures are saved to docs/figures for the documentation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt  


def plot_index_benchmark(results: list, path: Path) -> None:
    """Median query time and pages read vs table size, with and without an index (log-log)."""
    fig, (ax_time, ax_pages) = plt.subplots(1, 2, figsize=(11, 4.2))
    series = [(False, "No index (Seq Scan)", "o"), (True, "B-tree index", "s")]
    for indexed, label, marker in series:
        subset = [r for r in results if r.indexed == indexed]
        rows = [r.rows for r in subset]
        ax_time.plot(rows, [r.median_ms for r in subset], marker=marker, label=label)
        ax_pages.plot(rows, [r.pages for r in subset], marker=marker, label=label)
    for ax, ylabel in [(ax_time, "Median execution time (ms)"), (ax_pages, "Pages read (8 kB each)")]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Table size (rows)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Looking up one seller's items: Seq Scan grows with the table, the index stays flat")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    
def plot_category_heatmap(months: list, categories: list[str], matrix: list[list[float]], path: Path) -> None:
    """Heatmap of monthly revenue (columns) for the top categories (rows)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(categories)), labels=categories)
    ax.set_xticks(range(len(months)), labels=[m.strftime("%Y-%m") for m in months], rotation=90)
    fig.colorbar(image, ax=ax, label="Revenue (BRL)")
    ax.set_title("Monthly revenue of the top 10 categories (delivered orders)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_state_delivery(states: list[str], orders: list[int], late_pct: list[float],
                        avg_days: list[float], path: Path) -> None:
    """Average delivery time vs. late rate per state; marker area shows the number of orders."""
    fig, ax = plt.subplots(figsize=(8, 6))
    largest = max(orders)
    sizes = [20 + 600 * n / largest for n in orders]
    ax.scatter(avg_days, late_pct, s=sizes, alpha=0.6)
    for state, x, y in zip(states, avg_days, late_pct):
        ax.annotate(state, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.set_xlabel("Average delivery time (days)")
    ax.set_ylabel("Late deliveries (%)")
    ax.set_title("Delivery time vs. late rate per customer state (marker size = orders)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)