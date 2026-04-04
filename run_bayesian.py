"""Run Bayesian experiments with reduced grid and append to existing CSV.

Uses fewer draws/tune for speed, reduced dims and seeds.
Appends results to the existing benchmark_results.csv.
"""

import os
import sys
import time

import pandas as pd
from tqdm import tqdm

from src.experiments.runner import run_single_experiment
from src.experiments.grid import generate_experiment_list, generate_real_experiment_list

# ---- Reduced Bayesian grid ----
BAYESIAN_GRID = {
    "dataset": ["independent", "block_correlated", "toeplitz"],
    "model": ["horseshoe", "spike_slab"],
    "correlation_strength": [0.0, 0.3, 0.6, 0.9],
    "snr": [0.5, 1.0, 2.0, 5.0],
    "dim": [20, 50],            # skip dim=100 (too slow)
    "seeds": [42, 123, 456],    # 3 seeds instead of 5
}

# Override MCMC settings for speed
MCMC_OVERRIDES = {
    "n_samples": 1000,   # 1000 instead of 2000
    "n_tune": 500,       # 500 instead of 1000
    "n_chains": 2,
    "target_accept": 0.95,
}


def main():
    csv_path = os.path.join("results", "benchmark_results.csv")

    # Generate experiment list
    experiments = generate_experiment_list(BAYESIAN_GRID)

    # Add real dataset experiments
    real_exps = generate_real_experiment_list(
        models=["horseshoe", "spike_slab"],
        seeds=[42, 123, 456],
    )
    experiments += real_exps

    # Inject MCMC overrides
    for exp in experiments:
        exp.update(MCMC_OVERRIDES)

    total = len(experiments)
    print(f"\n{'='*60}")
    print(f"  Bayesian Experiments (reduced grid)")
    print(f"  Experiments: {total}")
    print(f"  MCMC: {MCMC_OVERRIDES['n_samples']} draws, {MCMC_OVERRIDES['n_tune']} tune")
    print(f"{'='*60}\n")

    # Set g++ path for PyTensor
    mingw_path = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft", "WinGet", "Packages",
        "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
        "mingw64", "bin",
    )
    if os.path.isdir(mingw_path):
        os.environ["PATH"] = mingw_path + os.pathsep + os.environ["PATH"]
        print(f"[OK] g++ found at {mingw_path}")
    else:
        print("[WARN] g++ not found, MCMC will be slow")

    # Run experiments
    results = []
    t_start = time.perf_counter()

    for i, exp in enumerate(tqdm(experiments, desc="Bayesian")):
        try:
            row = run_single_experiment(exp)
            results.append(row)
        except Exception as e:
            print(f"\n  [WARN] Failed: {exp.get('model')} {exp.get('dataset')} "
                  f"dim={exp.get('dim')} seed={exp.get('seed')} -- {e}")
            continue

        # Save intermediate results every 10 experiments
        if (i + 1) % 10 == 0:
            _save_partial(results, csv_path)

    elapsed = time.perf_counter() - t_start

    # Final save: append to existing CSV
    if results:
        _save_partial(results, csv_path)

    print(f"\n[OK] {len(results)}/{total} Bayesian experiments completed in {elapsed:.1f}s")


def _save_partial(new_results, csv_path):
    """Append new results to existing CSV."""
    df_new = pd.DataFrame(new_results)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        # Remove any existing Bayesian rows to avoid duplicates
        df_old = df_old[~df_old["model"].isin(["horseshoe", "spike_slab"])]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
