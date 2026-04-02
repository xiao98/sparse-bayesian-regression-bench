"""Abstract base class for all regression models in the benchmark.

Design adapted from BML-horseshoe-prior's BaseBayesianRegressor, expanded
to include fit/predict separation and optional posterior interval support.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseRegressor(ABC):
    """Base class for benchmark regressors.

    A regressor must implement:
      - fit(X, y)        : learn from training data
      - predict(X)       : point predictions
      - coef_            : estimated coefficient vector (property)

    Optionally override:
      - posterior_intervals(X, alpha) : credible/confidence intervals
    """

    def __init__(self, config) -> None:
        self.config = config
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on training data.

        Args:
            X: Feature matrix of shape (n, p).
            y: Target vector of shape (n,).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for new data.

        Args:
            X: Feature matrix of shape (m, p).

        Returns:
            Predictions of shape (m,).
        """

    @property
    @abstractmethod
    def coef_(self) -> np.ndarray:
        """Estimated coefficient vector of shape (p,)."""

    # ------------------------------------------------------------------
    # Optional: posterior / confidence intervals
    # ------------------------------------------------------------------

    @property
    def supports_intervals(self) -> bool:
        """Whether this model can produce posterior/confidence intervals."""
        return False

    def posterior_intervals(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (lower, upper) credible/confidence interval on coefficients.

        Args:
            X: Feature matrix (unused for coefficient intervals but kept
               for signature consistency with predictive intervals).
            alpha: Significance level (default 5 % → 95 % interval).

        Returns:
            Tuple of (lower, upper) arrays each of shape (p,),
            or None if not supported.
        """
        return None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def find_coefficients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Backward-compatible one-shot call (fit + return coef_).

        Mirrors the reference repo's regressor.find_coefficients() API.
        """
        self.fit(X, y)
        return self.coef_
