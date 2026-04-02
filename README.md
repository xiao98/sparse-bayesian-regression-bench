# Sparse Regression under Correlation and Weak Signals

**A Reproducible Benchmark of Classical and Bayesian Methods**

> Paper title: *"Sparse Regression under Correlation and Weak Signals: A Reproducible Benchmark of Classical and Bayesian Methods"*

## Overview

This repository benchmarks classical and Bayesian sparse regression methods under
challenging conditions: **correlated features**, **weak signals**, and **varying dimensionality**.
All results are saved to CSV and all figures are generated from saved data — ensuring full reproducibility.

### What is inherited (structure)

The benchmark engineering pattern is inspired by
[BML-horseshoe-prior](https://github.com/theosaulus/BML-horseshoe-prior):

| Inherited from reference | Our adaptation |
|---|---|
| `BaseDataset` abstract class (get_x_data, get_labels, get_beta) | Extended with get_x_test / get_y_test, posterior interval support |
| `BaseRegressor` abstract class (find_coefficients) | Extended to fit / predict / coef_ / posterior_intervals |
| Registry pattern (`name → Class` dict) | Same pattern in `src/data/__init__.py` and `src/models/__init__.py` |
| Hydra config organization | Same `configs/dataset/` + `configs/model/` structure |
| Benchmark runner loop | Replaced with grid-based runner + CSV persistence |

### What is original (content)

- **Synthetic data designs**: Independent Gaussian, block-correlated, Toeplitz covariance
- **Experiment axes**: correlation strength × SNR × dimensionality × random seeds
- **Metrics**: test MSE, coefficient L2 error, support P/R/F1, posterior coverage, seed stability
- **Compact real dataset** experiment (sklearn Diabetes)
- All results, tables, figures, and narrative

## Project Structure

```
sparse-bayesian-regression-bench/
├── run.py                          # Main entrypoint
├── requirements.txt
├── configs/
│   ├── config_default.yaml
│   ├── dataset/                    # Dataset configs (Hydra)
│   │   ├── independent.yaml
│   │   ├── block_correlated.yaml
│   │   ├── toeplitz.yaml
│   │   └── diabetes.yaml
│   └── model/                      # Model configs (Hydra)
│       ├── ols.yaml
│       ├── lasso.yaml
│       ├── ridge.yaml
│       ├── elastic_net.yaml
│       ├── horseshoe.yaml
│       └── spike_slab.yaml
├── src/
│   ├── data/                       # Data generation
│   │   ├── base_dataset.py         # Abstract base class
│   │   ├── synthetic.py            # 3 covariance designs
│   │   └── real_dataset.py         # Diabetes wrapper
│   ├── models/                     # Regressors
│   │   ├── base_model.py           # Abstract base class
│   │   ├── ols.py                  # OLS baseline
│   │   ├── lasso.py                # Lasso (L1 / Laplacian prior)
│   │   ├── ridge.py                # Ridge (L2 / Gaussian prior)
│   │   ├── elastic_net.py          # Elastic Net
│   │   ├── horseshoe.py            # Horseshoe (PyMC NUTS)
│   │   └── spike_slab.py           # Spike-and-Slab (PyMC)
│   ├── evaluation/
│   │   └── metrics.py              # Full metric suite
│   ├── experiments/
│   │   ├── grid.py                 # Experiment grid definition
│   │   └── runner.py               # Single-experiment orchestrator
│   └── visualization/
│       └── plots.py                # CSV → publication figures
├── results/
│   ├── figures/                    # Generated plots
│   └── tables/                     # Summary CSV tables
├── notebooks/                      # Analysis notebooks
└── report/                         # LaTeX report
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/sparse-bayesian-regression-bench.git
cd sparse-bayesian-regression-bench
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Usage

### Quick test (small grid, classical models only)
```bash
python run.py --mode quick
```

### Full benchmark (all synthetic datasets × all models)
```bash
python run.py --mode full
```

### Classical models only (skip slow Bayesian samplers)
```bash
python run.py --mode full --classical-only
```

### Include real dataset
```bash
python run.py --mode full --include-real
```

### Plot from saved CSV
```bash
python run.py --mode plot --csv results/benchmark_results.csv
```

## Models

| Model | Type | Prior interpretation | Posterior intervals |
|---|---|---|---|
| OLS | Classical | — (MLE) | ✗ |
| Lasso | Classical | Laplacian prior | ✗ |
| Ridge | Classical | Gaussian prior | ✗ |
| Elastic Net | Classical | L1 + L2 mix | ✗ |
| Horseshoe | Bayesian (MCMC) | Half-Cauchy shrinkage | ✓ |
| Spike-and-Slab | Bayesian (MCMC) | Bernoulli mixture | ✓ |

## Experiment Axes

| Axis | Values |
|---|---|
| Covariance design | Independent, Block-correlated, Toeplitz |
| Correlation strength (ρ) | 0.0, 0.3, 0.6, 0.9 |
| Signal-to-noise ratio | 0.5, 1.0, 2.0, 5.0 |
| Dimensionality (p) | 20, 50, 100 |
| Random seeds | 42, 123, 456, 789, 1024 |

## Metrics

- **Test MSE / RMSE** — prediction quality
- **Coefficient L2 error** — estimation accuracy
- **Support precision / recall / F1** — variable selection
- **95% posterior coverage** — Bayesian calibration
- **Average interval width** — uncertainty sharpness
- **Seed stability** (std across seeds) — reproducibility

## License

MIT
