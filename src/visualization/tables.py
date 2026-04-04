"""Generate LaTeX tables from benchmark CSV results.

Tables use booktabs formatting and are saved as standalone .tex fragments
ready to be \\input{} into a paper.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

TABLEDIR = os.path.join("results", "tables")


def _ensure_tabledir(tabledir: str = TABLEDIR) -> None:
    os.makedirs(tabledir, exist_ok=True)


def _bold_best(series: pd.Series, lower_is_better: bool = True) -> pd.Series:
    """Return formatted strings with the best value bolded."""
    if series.isna().all():
        return series.apply(lambda x: "---")
    best = series.min() if lower_is_better else series.max()
    def fmt(v):
        if pd.isna(v):
            return "---"
        s = f"{v:.3f}"
        if np.isclose(v, best):
            return f"\\textbf{{{s}}}"
        return s
    return series.apply(fmt)


MODEL_ORDER = ["ols", "ridge", "lasso", "elastic_net", "horseshoe", "spike_slab"]
MODEL_LABELS = {
    "ols": "OLS",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elastic_net": "Elastic Net",
    "horseshoe": "Horseshoe",
    "spike_slab": "Spike-and-Slab",
}


def _order_models(df: pd.DataFrame) -> pd.DataFrame:
    present = [m for m in MODEL_ORDER if m in df.index]
    return df.reindex(present)


def _rename_index(df: pd.DataFrame) -> pd.DataFrame:
    df.index = [MODEL_LABELS.get(m, m) for m in df.index]
    return df


def _write_tex(content: str, tabledir: str, filename: str) -> None:
    _ensure_tabledir(tabledir)
    path = os.path.join(tabledir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[tables] Saved {path}")


# ======================================================================
# 1. Grand summary: models x key metrics
# ======================================================================

def table_grand_summary(
    csv_path: str,
    tabledir: str = TABLEDIR,
    filename: str = "grand_summary.tex",
) -> None:
    df = pd.read_csv(csv_path)
    metrics = {
        "test_mse": ("Test MSE", True),
        "coef_l2_error": ("Coef. $L_2$ Error", True),
        "support_f1": ("Support $F_1$", False),
    }
    agg = df.groupby("model")

    rows = []
    for mdl in [m for m in MODEL_ORDER if m in df["model"].unique()]:
        grp = df[df["model"] == mdl]
        row = {"Model": MODEL_LABELS.get(mdl, mdl)}
        for col, (_, _) in metrics.items():
            if col in grp.columns and grp[col].notna().any():
                row[col + "_mean"] = grp[col].mean()
                row[col + "_std"] = grp[col].std()
            else:
                row[col + "_mean"] = np.nan
                row[col + "_std"] = np.nan
        rows.append(row)

    result = pd.DataFrame(rows).set_index("Model")

    # Format as mean +/- std
    lines = []
    lines.append(r"\begin{tabular}{l" + "c" * len(metrics) + "}")
    lines.append(r"\toprule")
    header = "Model"
    for col, (label, _) in metrics.items():
        header += f" & {label}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for _, row in result.iterrows():
        parts = [row.name]
        for col, (_, lower) in metrics.items():
            m = row[col + "_mean"]
            s = row[col + "_std"]
            if pd.isna(m):
                parts.append("---")
            else:
                parts.append(f"{m:.3f} $\\pm$ {s:.3f}")
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    _write_tex("\n".join(lines), tabledir, filename)


# ======================================================================
# 2. Correlation sensitivity: models x rho -> MSE
# ======================================================================

def table_correlation_sensitivity(
    csv_path: str,
    tabledir: str = TABLEDIR,
    filename: str = "correlation_sensitivity.tex",
) -> None:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["correlation_strength", "test_mse"])

    pivot = df.pivot_table(
        index="model", columns="correlation_strength",
        values="test_mse", aggfunc="mean",
    )
    pivot = _order_models(pivot)

    # Bold best per column
    for col in pivot.columns:
        pivot[col] = _bold_best(pivot[col], lower_is_better=True)

    pivot = _rename_index(pivot)
    pivot.columns = [f"$\\rho={c}$" for c in pivot.columns]

    tex = pivot.to_latex(escape=False, column_format="l" + "c" * len(pivot.columns))
    tex = tex.replace("\\toprule", "\\toprule").replace("\\midrule", "\\midrule")
    _write_tex(tex, tabledir, filename)


# ======================================================================
# 3. SNR sensitivity: models x SNR -> F1
# ======================================================================

def table_snr_sensitivity(
    csv_path: str,
    tabledir: str = TABLEDIR,
    filename: str = "snr_sensitivity.tex",
) -> None:
    df = pd.read_csv(csv_path)
    if "support_f1" not in df.columns:
        return
    df = df.dropna(subset=["snr", "support_f1"])

    pivot = df.pivot_table(
        index="model", columns="snr",
        values="support_f1", aggfunc="mean",
    )
    pivot = _order_models(pivot)

    for col in pivot.columns:
        pivot[col] = _bold_best(pivot[col], lower_is_better=False)

    pivot = _rename_index(pivot)
    pivot.columns = [f"SNR={c}" for c in pivot.columns]

    tex = pivot.to_latex(escape=False, column_format="l" + "c" * len(pivot.columns))
    _write_tex(tex, tabledir, filename)


# ======================================================================
# 4. Bayesian UQ: coverage + interval width
# ======================================================================

def table_bayesian_uq(
    csv_path: str,
    tabledir: str = TABLEDIR,
    filename: str = "bayesian_uq.tex",
) -> None:
    df = pd.read_csv(csv_path)
    if "coverage_95" not in df.columns:
        return
    df_b = df.dropna(subset=["coverage_95"])
    if df_b.empty:
        return

    agg = df_b.groupby("model").agg(
        coverage_mean=("coverage_95", "mean"),
        coverage_std=("coverage_95", "std"),
        width_mean=("avg_interval_width", "mean"),
        width_std=("avg_interval_width", "std"),
    )
    agg = _order_models(agg)
    agg = _rename_index(agg)

    lines = []
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Coverage (95\% HDI) & Avg.\ Interval Width \\")
    lines.append(r"\midrule")
    for mdl, row in agg.iterrows():
        cov = f"{row['coverage_mean']:.3f} $\\pm$ {row['coverage_std']:.3f}"
        wid = f"{row['width_mean']:.3f} $\\pm$ {row['width_std']:.3f}"
        lines.append(f"{mdl} & {cov} & {wid}" + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    _write_tex("\n".join(lines), tabledir, filename)


# ======================================================================
# 5. Runtime: models x dim
# ======================================================================

def table_runtime(
    csv_path: str,
    tabledir: str = TABLEDIR,
    filename: str = "runtime.tex",
) -> None:
    df = pd.read_csv(csv_path)
    if "fit_time_s" not in df.columns:
        return
    df = df.dropna(subset=["fit_time_s", "dim"])

    pivot = df.pivot_table(
        index="model", columns="dim",
        values="fit_time_s", aggfunc="mean",
    )
    pivot = _order_models(pivot)
    pivot = _rename_index(pivot)

    # Format with 2 decimal places
    for col in pivot.columns:
        pivot[col] = pivot[col].apply(lambda v: f"{v:.2f}" if not pd.isna(v) else "---")

    pivot.columns = [f"$p={int(c)}$" for c in pivot.columns]
    tex = pivot.to_latex(escape=False, column_format="l" + "c" * len(pivot.columns))
    _write_tex(tex, tabledir, filename)


# ======================================================================
# 6. Diabetes real-data results
# ======================================================================

def table_diabetes(
    csv_path: str,
    tabledir: str = TABLEDIR,
    filename: str = "diabetes.tex",
) -> None:
    df = pd.read_csv(csv_path)
    df_d = df[df["dataset"] == "diabetes"]
    if df_d.empty:
        return

    agg = df_d.groupby("model").agg(
        mse_mean=("test_mse", "mean"),
        mse_std=("test_mse", "std"),
        rmse_mean=("test_rmse", "mean"),
        rmse_std=("test_rmse", "std"),
        time_mean=("fit_time_s", "mean"),
    )
    agg = _order_models(agg)
    agg = _rename_index(agg)

    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Test MSE & Test RMSE & Time (s) \\")
    lines.append(r"\midrule")
    for mdl, row in agg.iterrows():
        mse = f"{row['mse_mean']:.3f} $\\pm$ {row['mse_std']:.3f}"
        rmse = f"{row['rmse_mean']:.3f} $\\pm$ {row['rmse_std']:.3f}"
        t = f"{row['time_mean']:.2f}"
        lines.append(f"{mdl} & {mse} & {rmse} & {t}" + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    _write_tex("\n".join(lines), tabledir, filename)


# ======================================================================
# Master generator
# ======================================================================

def generate_all_tables(csv_path: str, tabledir: str = TABLEDIR) -> None:
    """Generate all LaTeX tables from results CSV."""
    _ensure_tabledir(tabledir)
    print(f"[tables] Reading {csv_path} ...")
    table_grand_summary(csv_path, tabledir=tabledir)
    table_correlation_sensitivity(csv_path, tabledir=tabledir)
    table_snr_sensitivity(csv_path, tabledir=tabledir)
    table_bayesian_uq(csv_path, tabledir=tabledir)
    table_runtime(csv_path, tabledir=tabledir)
    table_diabetes(csv_path, tabledir=tabledir)
    print(f"[tables] All tables saved to {tabledir}/")
