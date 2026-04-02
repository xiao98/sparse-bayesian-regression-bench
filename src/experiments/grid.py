"""Experiment grid definition.

Defines the axes of the benchmark and generates the full Cartesian product
of experiment configurations to iterate over.
"""

from itertools import product
from typing import List, Dict, Any


# ======================================================================
# Default grid axes
# ======================================================================

DEFAULT_GRID: Dict[str, List[Any]] = {
    "dataset": ["independent", "block_correlated", "toeplitz"],
    "model": ["ols", "lasso", "ridge", "elastic_net", "horseshoe", "spike_slab"],
    "correlation_strength": [0.0, 0.3, 0.6, 0.9],
    "snr": [0.5, 1.0, 2.0, 5.0],
    "dim": [20, 50, 100],
    "seeds": [42, 123, 456, 789, 1024],
}

# Mapping from grid dataset tags to registry names
DATASET_TAG_TO_NAME = {
    "independent": "Independent Gaussian",
    "block_correlated": "Block Correlated",
    "toeplitz": "Toeplitz Correlated",
    "diabetes": "Diabetes",
}

# Mapping from grid model tags to registry names
MODEL_TAG_TO_NAME = {
    "ols": "OLS",
    "lasso": "Lasso",
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "horseshoe": "Horseshoe",
    "spike_slab": "Spike and Slab",
}


def generate_experiment_list(
    grid: Dict[str, List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate the full Cartesian product of experiment settings.

    Each element is a dict like:
        {
            "dataset": "block_correlated",
            "model": "lasso",
            "correlation_strength": 0.6,
            "snr": 2.0,
            "dim": 50,
            "seed": 42,
        }

    Note: for "independent" datasets correlation_strength is forced to 0.
    """
    if grid is None:
        grid = DEFAULT_GRID

    experiments = []
    for ds, mdl, rho, snr, dim, seed in product(
        grid["dataset"],
        grid["model"],
        grid["correlation_strength"],
        grid["snr"],
        grid["dim"],
        grid["seeds"],
    ):
        # Skip non-zero correlation for independent dataset
        if ds == "independent" and rho != 0.0:
            continue

        experiments.append({
            "dataset": ds,
            "model": mdl,
            "correlation_strength": rho,
            "snr": snr,
            "dim": dim,
            "seed": seed,
        })

    return experiments


def generate_real_experiment_list(
    models: List[str] = None,
    seeds: List[int] = None,
) -> List[Dict[str, Any]]:
    """Generate experiment list for the real (Diabetes) dataset."""
    if models is None:
        models = list(MODEL_TAG_TO_NAME.keys())
    if seeds is None:
        seeds = DEFAULT_GRID["seeds"]

    experiments = []
    for mdl, seed in product(models, seeds):
        experiments.append({
            "dataset": "diabetes",
            "model": mdl,
            "seed": seed,
        })
    return experiments
