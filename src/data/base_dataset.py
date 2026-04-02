"""Abstract base class for all datasets in the benchmark.

Design inherited from BML-horseshoe-prior's BaseDataset pattern:
each dataset exposes get_x_data(), get_labels(), get_beta() via a uniform
interface, plus test-set accessors added for our evaluation needs.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDataset(ABC):
    """Abstract base class for benchmark datasets.

    Every dataset must provide:
      - Training data  (X_train, y_train)
      - Test data      (X_test, y_test)
      - Ground-truth coefficients beta  (for synthetic datasets)

    Config is an OmegaConf DictConfig or plain dict.
    """

    def __init__(self, config) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Core abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_x_data(self) -> np.ndarray:
        """Return training features X_train of shape (n_train, p)."""

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        """Return training targets y_train of shape (n_train,)."""

    @abstractmethod
    def get_beta(self) -> np.ndarray:
        """Return ground-truth coefficient vector beta of shape (p,).

        For real datasets where beta is unknown, return a zero vector
        (metrics that depend on beta will be marked N/A).
        """

    @abstractmethod
    def get_x_test(self) -> np.ndarray:
        """Return test features X_test of shape (n_test, p)."""

    @abstractmethod
    def get_y_test(self) -> np.ndarray:
        """Return test targets y_test of shape (n_test,)."""

    # ------------------------------------------------------------------
    # Optional metadata
    # ------------------------------------------------------------------

    @property
    def has_true_beta(self) -> bool:
        """Whether ground-truth beta is available (synthetic = True)."""
        return True

    @property
    def sigma(self) -> Optional[float]:
        """Noise standard deviation, if known."""
        return getattr(self, "_sigma", None)

    # ------------------------------------------------------------------
    # Helper utilities (inspired by reference repo)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_sparse_beta(
        p: int,
        sparsity: float,
        coefficient_scale: float = 3.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a sparse coefficient vector.

        Args:
            p: Dimensionality.
            sparsity: Fraction of *zero* coefficients (0 = dense, 1 = all zero).
            coefficient_scale: Scale of non-zero entries (drawn ∼ N(0, scale²)).
            seed: Random seed for reproducibility.

        Returns:
            beta of shape (p,).
        """
        rng = np.random.default_rng(seed)
        beta = np.zeros(p)
        n_nonzero = max(1, int(p * (1.0 - sparsity)))
        indices = rng.choice(p, size=n_nonzero, replace=False)
        beta[indices] = rng.normal(0, coefficient_scale, size=n_nonzero)
        return beta

    @staticmethod
    def compute_total_variance(X: np.ndarray) -> float:
        """Sum of per-feature variances (diagnostic helper)."""
        return float(np.sum(np.var(X, axis=0)))
