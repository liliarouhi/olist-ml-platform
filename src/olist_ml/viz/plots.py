"""Plotting helpers. Figures are saved to docs/figures for the documentation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt  

import numpy as np


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
    
def plot_late_rate_panels(tables: dict, base_rate: float, path: Path) -> None:
    """2x2 bar charts of the late-delivery rate per group, with the overall rate as reference."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (title, table) in zip(axes.flat, tables.items()):
        ax.bar(range(len(table)), table["late_pct"], color="#4c72b0")
        ax.set_xticks(range(len(table)), labels=table.index.astype(str))
        ax.axhline(base_rate, color="grey", linestyle="--", label=f"overall {base_rate:.1f}%")
        ax.set_title(title)
        ax.set_ylabel("Late deliveries (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    fig.suptitle("Late-delivery rate by candidate feature")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation(df, columns: list[str], path: Path) -> None:
    """Annotated Spearman correlation matrix (rank-based, robust to skewed features)."""
    corr = df[columns].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), labels=columns, rotation=90)
    ax.set_yticks(range(len(columns)), labels=columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_title("Correlation between numeric features and the target")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    
def plot_loss_curve(history: list[float], path: Path,
    title: str = "Training loss: logistic regression (gradient descent)") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(history) + 1), history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted cross-entropy + L2")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    
def plot_nn_training(loss: list[float], val_pr_auc: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(loss) + 1)
    ax.plot(epochs, loss, color="#4c72b0", label="training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted cross-entropy + L2")
    if val_pr_auc:
        ax2 = ax.twinx()
        ax2.plot(range(1, len(val_pr_auc) + 1), val_pr_auc, color="#dd8452", label="validation PR-AUC")
        ax2.set_ylabel("Validation PR-AUC")
    ax.set_title("Neural network: Adam, early stopping on validation PR-AUC")
    fig.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(names: list[str], importances: np.ndarray, path: Path, top: int = 15) -> None:
    order = np.argsort(importances)[-top:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([names[i] for i in order], importances[order], color="#4c72b0")
    ax.set_xlabel("Share of total Gini decrease")
    ax.set_title(f"Decision tree: feature importance (top {top})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
def plot_curves(curves: dict, base_rate: float, path: Path) -> None:
    """ROC and precision-recall curves of several models on the same test set."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    for name, ((fpr, tpr), (precision, recall, _)) in curves.items():
        ax_roc.plot(fpr, tpr, label=name)
        ax_pr.plot(recall, precision, label=name)
    ax_roc.plot([0, 1], [0, 1], "k--", label="random")
    ax_pr.axhline(base_rate, color="k", linestyle="--", label=f"random ({base_rate:.3f})")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves (test)")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test)", ylim=(0, 1))
    for ax in (ax_roc, ax_pr):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bootstrap_intervals(table, base_rate: float, path: Path) -> None:
    """Test PR-AUC per model with 95% bootstrap confidence intervals."""
    t = table.sort_values("test_pr_auc")
    positions = list(range(len(t)))
    lower = np.maximum(t["test_pr_auc"] - t["test_pr_auc_low"], 0)
    upper = np.maximum(t["test_pr_auc_high"] - t["test_pr_auc"], 0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(t["test_pr_auc"], positions, xerr=[lower, upper], fmt="o", capsize=4)
    ax.set_yticks(positions, labels=t["model"])
    ax.axvline(base_rate, color="grey", linestyle="--", label=f"random ({base_rate:.3f})")
    ax.set_xlabel("Test PR-AUC with 95% bootstrap confidence interval")
    ax.set_title("Model comparison on the test set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)