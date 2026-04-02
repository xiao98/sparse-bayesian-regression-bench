"""Benchmark entrypoint: run the full sparse regression benchmark.

Usage:
    # Quick test: one dataset, one model
    python run.py --mode quick

    # Full benchmark grid (synthetic only)
    python run.py --mode full

    # Full benchmark + real dataset
    python run.py --mode full --include-real

    # Generate plots from existing CSV
    python run.py --mode plot --csv results/benchmark_results.csv

    # Classical models only (skip slow PyMC samplers)
    python run.py --mode full --classical-only
"""

import argparse
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

from src.experiments.grid import (
    generate_experiment_list,
    generate_real_experiment_list,
    DEFAULT_GRID,
)
from src.experiments.runner import run_single_experiment
from src.visualization.plots import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sparse Bayesian Regression Benchmark Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "plot"],
        default="quick",
        help="Run mode: 'quick' (small grid), 'full' (all axes), 'plot' (from CSV only).",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join("results", "benchmark_results.csv"),
        help="Path to results CSV (for plotting or saving).",
    )
    parser.add_argument(
        "--include-real",
        action="store_true",
        help="Include Diabetes real dataset experiments.",
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Skip Bayesian models (horseshoe, spike_slab) for faster runs.",
    )
    parser.add_argument(
        "--figdir",
        default=os.path.join("results", "figures"),
        help="Directory for output figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Plot-only mode ----
    if args.mode == "plot":
        if not os.path.exists(args.csv):
            print(f"Error: CSV not found at {args.csv}")
            sys.exit(1)
        generate_all_plots(args.csv, figdir=args.figdir)
        return

    # ---- Build experiment list ----
    if args.mode == "quick":
        grid = {
            "dataset": ["independent", "block_correlated"],
            "model": ["ols", "lasso", "ridge"],
            "correlation_strength": [0.0, 0.6],
            "snr": [1.0, 5.0],
            "dim": [20],
            "seeds": [42, 123],
        }
    else:
        grid = DEFAULT_GRID.copy()

    if args.classical_only:
        grid["model"] = [m for m in grid["model"]
                         if m not in ("horseshoe", "spike_slab")]

    experiments = generate_experiment_list(grid)
    if args.include_real:
        models = grid["model"]
        seeds = grid["seeds"]
        experiments += generate_real_experiment_list(models, seeds)

    total = len(experiments)
    print(f"\n{'='*60}")
    print(f"  Sparse Bayesian Regression Benchmark")
    print(f"  Mode: {args.mode} | Experiments: {total}")
    print(f"{'='*60}\n")

    # ---- Run experiments ----
    results = []
    t_start = time.perf_counter()

    for exp in tqdm(experiments, desc="Benchmark", disable=False):
        try:
            row = run_single_experiment(exp)
            results.append(row)
        except Exception as e:
            print(f"\n  [WARN] Failed: {exp} — {e}")
            continue

    elapsed = time.perf_counter() - t_start

    # ---- Save results ----
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.csv, index=False)
    print(f"\n[✓] {len(results)}/{total} experiments completed in {elapsed:.1f}s")
    print(f"[✓] Results saved to {args.csv}")

    # ---- Save summary table ----
    tables_dir = os.path.join("results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    if "test_mse" in df.columns:
        summary = df.groupby("model")["test_mse"].agg(["mean", "std"]).reset_index()
        summary.to_csv(os.path.join(tables_dir, "summary_mse.csv"), index=False)
        print(f"[✓] Summary table saved to {tables_dir}/summary_mse.csv")
        print("\n--- Test MSE Summary ---")
        print(summary.to_string(index=False))

    # ---- Generate plots ----
    generate_all_plots(args.csv, figdir=args.figdir)


if __name__ == "__main__":
    main()
