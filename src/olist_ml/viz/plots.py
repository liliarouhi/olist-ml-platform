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