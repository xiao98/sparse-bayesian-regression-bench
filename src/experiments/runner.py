"""Single-experiment runner.

Encapsulates: create dataset → fit model → evaluate metrics → return row dict.
Designed to be called in a loop by the main benchmark runner (run.py).
"""

from typing import Dict, Any
import time

import numpy as np

from src.data import dataset_name_to_class
from src.models import model_name_to_class
from src.evaluation.metrics import compute_all_metrics
from src.experiments.grid import DATASET_TAG_TO_NAME, MODEL_TAG_TO_NAME


def run_single_experiment(exp: Dict[str, Any]) -> Dict[str, Any]:
    """Run one (dataset, model, seed, hyperparams) experiment.

    Args:
        exp: Experiment specification dict with keys:
            dataset, model, seed, dim, snr, correlation_strength, etc.

    Returns:
        Flat dict of experiment identifiers + metric values,
        ready to be appended to a results DataFrame.
    """
    ds_tag = exp["dataset"]
    mdl_tag = exp["model"]
    seed = exp["seed"]

    # ---- Build dataset config ----
    ds_name = DATASET_TAG_TO_NAME[ds_tag]
    ds_config: Dict[str, Any] = {"seed": seed}

    if ds_tag == "diabetes":
        # Real dataset — minimal config
        ds_config["test_size"] = exp.get("test_size", 0.2)
    else:
        # Synthetic datasets
        ds_config["dim"] = exp["dim"]
        ds_config["n_train"] = exp.get("n_train", max(50, int(exp["dim"] * 1.5)))
        ds_config["n_test"] = exp.get("n_test", 200)
        ds_config["snr"] = exp["snr"]
        ds_config["sparsity"] = exp.get("sparsity", 0.8)
        ds_config["coefficient_scale"] = exp.get("coefficient_scale", 3.0)
        if ds_tag in ("block_correlated", "toeplitz"):
            ds_config["correlation_strength"] = exp["correlation_strength"]
        if ds_tag == "block_correlated":
            ds_config["block_size"] = exp.get("block_size", 5)

    # ---- Create dataset ----
    DatasetClass = dataset_name_to_class[ds_name]
    dataset = DatasetClass(ds_config)

    X_train = dataset.get_x_data()
    y_train = dataset.get_labels()
    X_test = dataset.get_x_test()
    y_test = dataset.get_y_test()
    beta_true = dataset.get_beta()
    has_true_beta = dataset.has_true_beta

    # ---- Build model config ----
    mdl_name = MODEL_TAG_TO_NAME[mdl_tag]
    mdl_config: Dict[str, Any] = {"seed": seed}
    # Merge any model-specific overrides from exp
    for key in ("n_samples", "n_tune", "n_chains", "target_accept",
                "tau0_scale", "slab_scale", "spike_scale", "inclusion_prob",
                "n_alphas", "cv", "max_iter"):
        if key in exp:
            mdl_config[key] = exp[key]

    # ---- Fit model ----
    ModelClass = model_name_to_class[mdl_name]
    model = ModelClass(mdl_config)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # ---- Predict ----
    y_pred = model.predict(X_test)
    beta_hat = model.coef_

    # ---- Posterior intervals (if supported) ----
    intervals = None
    if model.supports_intervals and has_true_beta:
        intervals = model.posterior_intervals(X_test, alpha=0.05)

    # ---- Evaluate ----
    metrics = compute_all_metrics(
        y_test, y_pred, beta_true, beta_hat,
        has_true_beta=has_true_beta,
        intervals=intervals,
    )

    # ---- Assemble result row ----
    row: Dict[str, Any] = {
        "dataset": ds_tag,
        "model": mdl_tag,
        "seed": seed,
        "fit_time_s": round(fit_time, 4),
    }
    # Add experiment axes (for synthetic)
    if ds_tag != "diabetes":
        row["dim"] = exp["dim"]
        row["snr"] = exp["snr"]
        row["correlation_strength"] = exp.get("correlation_strength", 0.0)

    row.update(metrics)
    return row
