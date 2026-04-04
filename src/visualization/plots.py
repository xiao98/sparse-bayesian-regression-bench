"""Visualization module: generate publication-quality plots from CSV results.

Every plot reads from a saved CSV — no in-memory results needed.
Figures are saved to results/figures/ in both PDF and PNG formats.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
})
sns.set_theme(style="whitegrid", font_scale=1.05)

FIGDIR = os.path.join("results", "figures")

# Consistent ordering, labels, and colours across all figures
MODEL_ORDER = ["ols", "ridge", "lasso", "elastic_net", "horseshoe", "spike_slab"]
MODEL_LABELS = {
    "ols": "OLS",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elastic_net": "Elastic Net",
    "horseshoe": "Horseshoe",
    "spike_slab": "Spike-and-Slab",
}
# Colourblind-friendly palette (paired style)
MODEL_COLORS = {
    "ols": "#999999",
    "ridge": "#E69F00",
    "lasso": "#56B4E9",
    "elastic_net": "#009E73",
    "horseshoe": "#D55E00",
    "spike_slab": "#CC79A7",
}

DATASET_LABELS = {
    "independent": "Independent",
    "block_correlated": "Block Correlated",
    "toeplitz": "Toeplitz",
    "diabetes": "Diabetes",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_figdir(figdir: str = FIGDIR) -> None:
    os.makedirs(figdir, exist_ok=True)


def _save_fig(fig, figdir: str, basename: str) -> None:
    """Save figure as both PDF and PNG."""
    _ensure_figdir(figdir)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(figdir, f"{basename}.{ext}"),
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)


def _ordered_models(df: pd.DataFrame):
    """Return model list in canonical order, filtered to those present in df."""
    present = df["model"].unique()
    return [m for m in MODEL_ORDER if m in present]


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, "#333333")


# ======================================================================
# 1. Heatmap: Model x Correlation -> MSE
# ======================================================================

def plot_mse_heatmap(
    csv_path: str,
    metric: str = "test_mse",
    figdir: str = FIGDIR,
    basename: str = "heatmap_mse_vs_correlation",
) -> None:
    """Heatmap of average test MSE for each (model, correlation_strength)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[metric, "correlation_strength"])

    models = _ordered_models(df)
    df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)

    pivot = df.pivot_table(
        index="model", columns="correlation_strength",
        values=metric, aggfunc="mean",
    ).reindex(models)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
        yticklabels=[_label(m) for m in pivot.index],
    )
    ax.set_title(f"Average Test MSE by Model and Correlation Strength")
    ax.set_ylabel("")
    ax.set_xlabel(r"Correlation Strength ($\rho$)")
    _save_fig(fig, figdir, basename)


# ======================================================================
# 2. Line plot: SNR vs F1 for each model
# ======================================================================

def plot_f1_vs_snr(
    csv_path: str,
    figdir: str = FIGDIR,
    basename: str = "lineplot_f1_vs_snr",
) -> None:
    """Line plot of support F1 vs SNR, one line per model."""
    df = pd.read_csv(csv_path)
    if "support_f1" not in df.columns:
        return

    df = df.dropna(subset=["support_f1", "snr"])
    models = _ordered_models(df)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mdl in models:
        grp = df[df["model"] == mdl]
        summary = grp.groupby("snr")["support_f1"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            summary["snr"], summary["mean"], yerr=summary["std"],
            marker="o", capsize=3, label=_label(mdl), color=_color(mdl), linewidth=1.5,
        )
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax.set_ylabel("Support F1 Score")
    ax.set_title("Variable Selection Performance vs Signal Strength")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    _save_fig(fig, figdir, basename)


# ======================================================================
# 3. Box plot: seed stability
# ======================================================================

def plot_seed_stability(
    csv_path: str,
    metric: str = "test_mse",
    figdir: str = FIGDIR,
    basename: str = "boxplot_seed_stability",
) -> None:
    """Box plot showing metric spread across seeds for each model."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[metric])
    models = _ordered_models(df)
    df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)
    df = df.sort_values("model")

    palette = [_color(m) for m in models]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(
        data=df, x="model", y=metric, ax=ax, palette=palette, order=models,
    )
    ax.set_xticklabels([_label(m) for m in models], rotation=25, ha="right")
    ax.set_title(f"Test MSE Distribution Across Random Seeds")
    ax.set_xlabel("")
    ax.set_ylabel("Test MSE")
    _save_fig(fig, figdir, basename)


# ======================================================================
# 4. Bar chart: posterior coverage
# ======================================================================

def plot_coverage_comparison(
    csv_path: str,
    figdir: str = FIGDIR,
    basename: str = "barchart_coverage",
) -> None:
    """Bar chart comparing 95 pct posterior coverage across Bayesian models."""
    df = pd.read_csv(csv_path)
    if "coverage_95" not in df.columns:
        return
    df_cov = df.dropna(subset=["coverage_95"])
    if df_cov.empty:
        return

    models = [m for m in _ordered_models(df_cov)]
    summary = df_cov.groupby("model")["coverage_95"].agg(["mean", "std"]).reindex(models).dropna()

    fig, ax = plt.subplots(figsize=(5, 4))
    x = range(len(summary))
    bars = ax.bar(
        x, summary["mean"], yerr=summary["std"], capsize=5,
        color=[_color(m) for m in summary.index],
        edgecolor="black", linewidth=0.5,
    )
    ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1, label="Nominal 95%")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_label(m) for m in summary.index], rotation=15, ha="right")
    ax.set_ylabel("Coverage Probability")
    ax.set_title("95% Posterior Interval Coverage")
    ax.set_ylim(0, 1.05)
    ax.legend()
    _save_fig(fig, figdir, basename)


# ======================================================================
# 5. Coefficient L2 error heatmap: Model x Dimensionality
# ======================================================================

def plot_l2_vs_dim(
    csv_path: str,
    figdir: str = FIGDIR,
    basename: str = "heatmap_l2_vs_dim",
) -> None:
    """Heatmap of coefficient L2 error by model x dimensionality."""
    df = pd.read_csv(csv_path)
    if "coef_l2_error" not in df.columns:
        return
    df = df.dropna(subset=["coef_l2_error", "dim"])

    models = _ordered_models(df)
    pivot = df.pivot_table(
        index="model", columns="dim", values="coef_l2_error", aggfunc="mean",
    ).reindex(models)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="Blues", ax=ax,
        yticklabels=[_label(m) for m in pivot.index],
    )
    ax.set_title("Coefficient L2 Error by Model and Dimensionality")
    ax.set_ylabel("")
    ax.set_xlabel("Dimensionality (p)")
    _save_fig(fig, figdir, basename)


# ======================================================================
# 6. NEW — MSE vs SNR by dataset (hero figure, 3-panel)
# ======================================================================

def plot_mse_vs_snr_by_dataset(
    csv_path: str,
    figdir: str = FIGDIR,
    basename: str = "lineplot_mse_vs_snr_by_dataset",
) -> None:
    """Three-panel line plot: test MSE vs SNR, one panel per synthetic dataset."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["test_mse", "snr"])
    datasets = [d for d in ["independent", "block_correlated", "toeplitz"]
                if d in df["dataset"].unique()]
    if not datasets:
        return

    models = _ordered_models(df)
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.2), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for mdl in models:
            grp = sub[sub["model"] == mdl]
            if grp.empty:
                continue
            summary = grp.groupby("snr")["test_mse"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                summary["snr"], summary["mean"], yerr=summary["std"],
                marker="o", capsize=3, label=_label(mdl),
                color=_color(mdl), linewidth=1.5,
            )
        ax.set_xscale("log")
        ax.set_xlabel("SNR")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Test MSE")
    axes[-1].legend(loc="upper right", fontsize=7, ncol=1)
    fig.suptitle("Prediction Error vs Signal-to-Noise Ratio", y=1.02, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, figdir, basename)


# ======================================================================
# 7. NEW — MSE degradation with correlation (2-panel)
# ======================================================================

def plot_mse_degradation_with_correlation(
    csv_path: str,
    figdir: str = FIGDIR,
    basename: str = "lineplot_mse_vs_correlation",
) -> None:
    """Two-panel line plot: MSE vs rho for Block Correlated and Toeplitz."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["test_mse", "correlation_strength"])
    datasets = [d for d in ["block_correlated", "toeplitz"]
                if d in df["dataset"].unique()]
    if not datasets:
        return

    models = _ordered_models(df)
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.2), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for mdl in models:
            grp = sub[sub["model"] == mdl]
            if grp.empty:
                continue
            summary = grp.groupby("correlation_strength")["test_mse"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                summary["correlation_strength"], summary["mean"], yerr=summary["std"],
                marker="s", capsize=3, label=_label(mdl),
                color=_color(mdl), linewidth=1.5,
            )
        ax.set_xlabel(r"Correlation Strength ($\rho$)")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Test MSE")
    axes[-1].legend(loc="upper left", fontsize=7)
    fig.suptitle("Prediction Error Degradation with Feature Correlation", y=1.02, fontsize=13)
    fig.tight_layout()
    _save_fig(fig, figdir, basename)


# ======================================================================
# 8. NEW — Runtime comparison (bar chart, log scale)
# ======================================================================

def plot_runtime_comparison(
    csv_path: str,
    figdir: str = FIGDIR,
    basename: str = "barchart_runtime",
) -> None:
    """Bar chart of mean fit time per model (log scale)."""
    df = pd.read_csv(csv_path)
    if "fit_time_s" not in df.columns:
        return
    df = df.dropna(subset=["fit_time_s"])

    models = _ordered_models(df)
    summary = df.groupby("model")["fit_time_s"].agg(["mean", "std"]).reindex(models).dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(summary))
    ax.bar(
        x, summary["mean"], yerr=summary["std"], capsize=4,
        color=[_color(m) for m in summary.index],
        edgecolor="black", linewidth=0.5,
    )
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_label(m) for m in summary.index], rotation=20, ha="right")
    ax.set_ylabel("Fit Time (seconds, log scale)")
    ax.set_title("Computational Cost Comparison")
    _save_fig(fig, figdir, basename)


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
    plot_mse_vs_snr_by_dataset(csv_path, figdir=figdir)
    plot_mse_degradation_with_correlation(csv_path, figdir=figdir)
    plot_runtime_comparison(csv_path, figdir=figdir)
    print(f"[plots] All figures saved to {figdir}/")
