"""Synthetic datasets with three covariance structures.

Designs:
  1. IndependentGaussian   — Σ = I  (baseline / control)
  2. BlockCorrelated       — block-diagonal with intra-block correlation ρ
  3. ToeplitzCorrelated    — Σ[i,j] = ρ^|i-j|  (AR(1)-like)

Each generator supports axes:
  - correlation_strength (ρ)
  - signal-to-noise ratio (SNR)
  - dimensionality (p)
  - random seed
"""

from typing import Optional

import numpy as np
from scipy.linalg import toeplitz

from src.data.base_dataset import BaseDataset


# ======================================================================
# Helpers
# ======================================================================

def _generate_data(
    cov_matrix: np.ndarray,
    beta: np.ndarray,
    n_train: int,
    n_test: int,
    snr: float,
    rng: np.random.Generator,
) -> tuple:
    """Generate (X_train, y_train, X_test, y_test, sigma) from Σ, β, SNR.

    SNR = Var(Xβ) / σ²  →  σ² = Var(Xβ) / SNR
    We first draw X, compute empirical signal variance, then set σ accordingly.
    """
    p = len(beta)
    n_total = n_train + n_test
    X = rng.multivariate_normal(np.zeros(p), cov_matrix, size=n_total)

    signal = X @ beta
    signal_var = np.var(signal)
    # Avoid degenerate case where signal_var == 0
    sigma = np.sqrt(max(signal_var, 1e-8) / snr)
    noise = rng.normal(0, sigma, size=n_total)
    y = signal + noise

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, y_train, X_test, y_test, sigma


# ======================================================================
# 1. Independent Gaussian
# ======================================================================

class IndependentGaussianDataset(BaseDataset):
    """Features drawn from N(0, I).  Serves as the control condition."""

    def __init__(self, config) -> None:
        super().__init__(config)
        cfg = config
        p = int(cfg["dim"])
        n_train = int(cfg["n_train"])
        n_test = int(cfg.get("n_test", 200))
        snr = float(cfg.get("snr", 2.0))
        sparsity = float(cfg.get("sparsity", 0.8))
        coeff_scale = float(cfg.get("coefficient_scale", 3.0))
        seed = int(cfg.get("seed", 42))

        rng = np.random.default_rng(seed)
        self._beta = self.generate_sparse_beta(p, sparsity, coeff_scale, seed)
        cov = np.eye(p)

        (
            self._X_train, self._y_train,
            self._X_test, self._y_test,
            self._sigma,
        ) = _generate_data(cov, self._beta, n_train, n_test, snr, rng)

    def get_x_data(self) -> np.ndarray:
        return self._X_train

    def get_labels(self) -> np.ndarray:
        return self._y_train

    def get_beta(self) -> np.ndarray:
        return self._beta

    def get_x_test(self) -> np.ndarray:
        return self._X_test

    def get_y_test(self) -> np.ndarray:
        return self._y_test


# ======================================================================
# 2. Block Correlated
# ======================================================================

class BlockCorrelatedDataset(BaseDataset):
    """Features with block-diagonal covariance.

    Blocks of size `block_size`; within each block Σ[i,j] = ρ for i≠j, 1 on diagonal.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        cfg = config
        p = int(cfg["dim"])
        n_train = int(cfg["n_train"])
        n_test = int(cfg.get("n_test", 200))
        snr = float(cfg.get("snr", 2.0))
        sparsity = float(cfg.get("sparsity", 0.8))
        coeff_scale = float(cfg.get("coefficient_scale", 3.0))
        rho = float(cfg.get("correlation_strength", 0.6))
        block_size = int(cfg.get("block_size", 5))
        seed = int(cfg.get("seed", 42))

        rng = np.random.default_rng(seed)
        self._beta = self.generate_sparse_beta(p, sparsity, coeff_scale, seed)

        # Build block-diagonal covariance
        cov = np.eye(p)
        n_blocks = p // block_size
        for b in range(n_blocks):
            s = b * block_size
            e = s + block_size
            block = np.full((block_size, block_size), rho)
            np.fill_diagonal(block, 1.0)
            cov[s:e, s:e] = block
        # Remaining features stay independent

        (
            self._X_train, self._y_train,
            self._X_test, self._y_test,
            self._sigma,
        ) = _generate_data(cov, self._beta, n_train, n_test, snr, rng)

    def get_x_data(self) -> np.ndarray:
        return self._X_train

    def get_labels(self) -> np.ndarray:
        return self._y_train

    def get_beta(self) -> np.ndarray:
        return self._beta

    def get_x_test(self) -> np.ndarray:
        return self._X_test

    def get_y_test(self) -> np.ndarray:
        return self._y_test


# ======================================================================
# 3. Toeplitz Correlated
# ======================================================================

class ToeplitzCorrelatedDataset(BaseDataset):
    """Features with Toeplitz (AR(1)-like) covariance: Σ[i,j] = ρ^|i−j|."""

    def __init__(self, config) -> None:
        super().__init__(config)
        cfg = config
        p = int(cfg["dim"])
        n_train = int(cfg["n_train"])
        n_test = int(cfg.get("n_test", 200))
        snr = float(cfg.get("snr", 2.0))
        sparsity = float(cfg.get("sparsity", 0.8))
        coeff_scale = float(cfg.get("coefficient_scale", 3.0))
        rho = float(cfg.get("correlation_strength", 0.6))
        seed = int(cfg.get("seed", 42))

        rng = np.random.default_rng(seed)
        self._beta = self.generate_sparse_beta(p, sparsity, coeff_scale, seed)

        first_row = rho ** np.arange(p)
        cov = toeplitz(first_row)

        (
            self._X_train, self._y_train,
            self._X_test, self._y_test,
            self._sigma,
        ) = _generate_data(cov, self._beta, n_train, n_test, snr, rng)

    def get_x_data(self) -> np.ndarray:
        return self._X_train

    def get_labels(self) -> np.ndarray:
        return self._y_train

    def get_beta(self) -> np.ndarray:
        return self._beta

    def get_x_test(self) -> np.ndarray:
        return self._X_test

    def get_y_test(self) -> np.ndarray:
        return self._y_test
