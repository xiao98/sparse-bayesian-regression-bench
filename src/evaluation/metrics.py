"""Evaluation metrics for sparse regression benchmarks.

Metrics:
  - Test MSE / RMSE
  - Coefficient L2 error
  - Support recovery: precision, recall, F1
  - Posterior interval coverage (for Bayesian models)
  - Seed stability (standard deviation across seeds)
"""

from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ======================================================================
# Prediction quality
# ======================================================================

def test_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error on test set."""
    return float(np.mean((y_true - y_pred) ** 2))


def test_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error on test set."""
    return float(np.sqrt(test_mse(y_true, y_pred)))


# ======================================================================
# Coefficient estimation
# ======================================================================

def coefficient_l2_error(beta_true: np.ndarray, beta_hat: np.ndarray) -> float:
    """L2 (Euclidean) error between true and estimated coefficients."""
    return float(np.sqrt(np.sum((beta_true - beta_hat) ** 2)))


def coefficient_mse(beta_true: np.ndarray, beta_hat: np.ndarray) -> float:
    """Mean Squared Error of coefficient estimates."""
    return float(np.mean((beta_true - beta_hat) ** 2))


# ======================================================================
# Support recovery (variable selection)
# ======================================================================

def _support_masks(
    beta_true: np.ndarray,
    beta_hat: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary masks: True where coefficient is non-zero."""
    true_support = np.abs(beta_true) > threshold
    hat_support = np.abs(beta_hat) > threshold
    return true_support, hat_support


def support_precision(
    beta_true: np.ndarray,
    beta_hat: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """Precision of support recovery: TP / (TP + FP)."""
    true_s, hat_s = _support_masks(beta_true, beta_hat, threshold)
    tp = np.sum(true_s & hat_s)
    fp = np.sum(~true_s & hat_s)
    return float(tp / max(tp + fp, 1))


def support_recall(
    beta_true: np.ndarray,
    beta_hat: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """Recall of support recovery: TP / (TP + FN)."""
    true_s, hat_s = _support_masks(beta_true, beta_hat, threshold)
    tp = np.sum(true_s & hat_s)
    fn = np.sum(true_s & ~hat_s)
    return float(tp / max(tp + fn, 1))


def support_f1(
    beta_true: np.ndarray,
    beta_hat: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """F1 score of support recovery."""
    p = support_precision(beta_true, beta_hat, threshold)
    r = support_recall(beta_true, beta_hat, threshold)
    return float(2 * p * r / max(p + r, 1e-12))


# ======================================================================
# Uncertainty quantification
# ======================================================================

def posterior_coverage(
    beta_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of true coefficients within [lower, upper] interval.

    A well-calibrated 95 % interval should achieve ~0.95 coverage.
    """
    inside = (beta_true >= lower) & (beta_true <= upper)
    return float(np.mean(inside))


def average_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of posterior/confidence intervals."""
    return float(np.mean(upper - lower))


# ======================================================================
# Seed stability
# ======================================================================

def seed_stability(
    results_df: pd.DataFrame,
    metric_col: str,
    group_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Compute std of a metric across seeds.

    Args:
        results_df: Full results table with a 'seed' column.
        metric_col: Name of the metric column to aggregate.
        group_cols: Columns to group by (default: ['dataset', 'model']).

    Returns:
        DataFrame with mean and std of the metric.
    """
    if group_cols is None:
        group_cols = ["dataset", "model"]
    return (
        results_df
        .groupby(group_cols)[metric_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": f"{metric_col}_mean", "std": f"{metric_col}_std"})
    )


# ======================================================================
# All-in-one evaluation
# ======================================================================

def compute_all_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    beta_true: np.ndarray,
    beta_hat: np.ndarray,
    has_true_beta: bool = True,
    intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    threshold: float = 0.01,
) -> Dict[str, float]:
    """Compute the full metric suite for one (dataset, model, seed) run.

    Returns a flat dict ready for DataFrame row conversion.
    """
    result: Dict[str, float] = {}

    # Prediction
    result["test_mse"] = test_mse(y_test, y_pred)
    result["test_rmse"] = test_rmse(y_test, y_pred)

    # Coefficient estimation
    if has_true_beta:
        result["coef_l2_error"] = coefficient_l2_error(beta_true, beta_hat)
        result["coef_mse"] = coefficient_mse(beta_true, beta_hat)

        # Support recovery
        result["support_precision"] = support_precision(beta_true, beta_hat, threshold)
        result["support_recall"] = support_recall(beta_true, beta_hat, threshold)
        result["support_f1"] = support_f1(beta_true, beta_hat, threshold)

    # Posterior coverage
    if intervals is not None and has_true_beta:
        lower, upper = intervals
        result["coverage_95"] = posterior_coverage(beta_true, lower, upper)
        result["avg_interval_width"] = average_interval_width(lower, upper)

    return result
