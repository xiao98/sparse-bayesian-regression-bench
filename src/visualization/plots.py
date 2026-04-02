"""Visualization module: generate publication-quality plots from CSV results.

Every plot reads from a saved CSV — no in-memory results needed.
Figures are saved to results/figures/.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGDIR = os.path.join("results", "figures")


def _ensure_figdir(figdir: str = FIGDIR) -> None:
    os.makedirs(figdir, exist_ok=True)


# ======================================================================
# 1. Heatmap: Model × Correlation → MSE
# ======================================================================

def plot_mse_heatmap(
    csv_path: str,
    metric: str = "test_mse",
    figdir: str = FIGDIR,
    filename: str = "heatmap_mse_vs_correlation.png",
) -> None:
    """Heatmap of average test MSE for each (model, correlation_strength)."""
    _ensure_figdir(figdir)
    df = pd.read_csv(csv_path)

    pivot = df.pivot_table(
        index="model",
        columns="correlation_strength",
        values=metric,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title(f"Average {metric} by Model × Correlation Strength")
    ax.set_ylabel("Model")
    ax.set_xlabel("Correlation Strength (ρ)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, filename), dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================
# 2. Line plot: SNR vs F1 for each model
# ======================================================================

def plot_f1_vs_snr(
    csv_path: str,
    figdir: str = FIGDIR,
    filename: str = "lineplot_f1_vs_snr.png",
) -> None:
    """Line plot of support F1 vs SNR, one line per model."""
    _ensure_figdir(figdir)
    df = pd.read_csv(csv_path)

    if "support_f1" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, grp in df.groupby("model"):
        summary = grp.groupby("snr")["support_f1"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            summary["snr"], summary["mean"], yerr=summary["std"],
            marker="o", capsize=3, label=model_name,
        )
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax.set_ylabel("Support F1 Score")
    ax.set_title("Variable Selection F1 vs SNR")
    ax.legend(loc="best", fontsize=9)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, filename), dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================
# 3. Box plot: seed stability
# ======================================================================

def plot_seed_stability(
    csv_path: str,
    metric: str = "test_mse",
    figdir: str = FIGDIR,
    filename: str = "boxplot_seed_stability.png",
) -> None:
    """Box plot showing metric spread across seeds for each model."""
    _ensure_figdir(figdir)
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="model", y=metric, ax=ax, palette="Set2")
    ax.set_title(f"{metric} Distribution Across Seeds")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, filename), dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================
# 4. Bar chart: posterior coverage
# ======================================================================

def plot_coverage_comparison(
    csv_path: str,
    figdir: str = FIGDIR,
    filename: str = "barchart_coverage.png",
) -> None:
    """Bar chart comparing 95% posterior coverage across Bayesian models."""
    _ensure_figdir(figdir)
    df = pd.read_csv(csv_path)

    if "coverage_95" not in df.columns:
        return

    df_cov = df.dropna(subset=["coverage_95"])
    if df_cov.empty:
        return

    summary = df_cov.groupby("model")["coverage_95"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        summary["model"], summary["mean"],
        yerr=summary["std"], capsize=4, color=sns.color_palette("muted"),
        edgecolor="black", linewidth=0.5,
    )
    ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1, label="Nominal 95%")
    ax.set_ylabel("Coverage")
    ax.set_title("95% Posterior Interval Coverage")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, filename), dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================
# 5. Coefficient L2 error heatmap: Model × Dimensionality
# ======================================================================

def plot_l2_vs_dim(
    csv_path: str,
    figdir: str = FIGDIR,
    filename: str = "heatmap_l2_vs_dim.png",
) -> None:
    """Heatmap of coefficient L2 error by model × dimensionality."""
    _ensure_figdir(figdir)
    df = pd.read_csv(csv_path)

    if "coef_l2_error" not in df.columns:
        return

    pivot = df.pivot_table(
        index="model", columns="dim", values="coef_l2_error", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Coefficient L2 Error by Model × Dimensionality")
    ax.set_ylabel("Model")
    ax.set_xlabel("Dimensionality (p)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, filename), dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================
# Master generator
# ======================================================================

def generate_all_plots(csv_path: str, figdir: str = FIGDIR) -> None:
    """Generate all standard benchmark figures from results CSV."""
    _ensure_figdir(figdir)
    print(f"[plots] Reading {csv_path} ...")
    plot_mse_heatmap(csv_path, figdir=figdir)
    plot_f1_vs_snr(csv_path, figdir=figdir)
    plot_seed_stability(csv_path, figdir=figdir)
    plot_coverage_comparison(csv_path, figdir=figdir)
    plot_l2_vs_dim(csv_path, figdir=figdir)
    print(f"[plots] All figures saved to {figdir}/")
