"""Horseshoe prior regressor via PyMC (NUTS sampler).

The horseshoe prior (Carvalho et al., 2009) places:
    β_j | λ_j, τ  ~  N(0, λ_j² τ²)
    λ_j            ~  C⁺(0, 1)      (half-Cauchy local shrinkage)
    τ              ~  C⁺(0, τ₀)     (half-Cauchy global shrinkage)

This implementation provides posterior credible intervals for β.
"""

from typing import Optional, Tuple

import numpy as np

from src.models.base_model import BaseRegressor

try:
    import pymc as pm
    import arviz as az

    _HAS_PYMC = True
except ImportError:
    _HAS_PYMC = False


class HorseshoeRegressor(BaseRegressor):
    """Horseshoe prior fitted with NUTS via PyMC."""

    def __init__(self, config) -> None:
        super().__init__(config)
        if not _HAS_PYMC:
            raise ImportError("PyMC is required for HorseshoeRegressor. "
                              "Install via: pip install pymc arviz")
        self.n_samples = int(config.get("n_samples", 2000))
        self.n_tune = int(config.get("n_tune", 1000))
        self.n_chains = int(config.get("n_chains", 2))
        self.target_accept = float(config.get("target_accept", 0.95))
        self.tau0_scale = float(config.get("tau0_scale", 1.0))
        self.seed = int(config.get("seed", 42))
        self._coef = None
        self._trace = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p = X.shape

        with pm.Model() as model:
            # Global shrinkage
            tau = pm.HalfCauchy("tau", beta=self.tau0_scale)
            # Local shrinkage
            lam = pm.HalfCauchy("lam", beta=1.0, shape=p)
            # Coefficients
            beta = pm.Normal("beta", mu=0, sigma=lam * tau, shape=p)
            # Noise
            sigma = pm.HalfCauchy("sigma", beta=2.0)
            # Likelihood
            mu = pm.math.dot(X, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Sample
            self._trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                target_accept=self.target_accept,
                random_seed=self.seed,
                progressbar=False,
                return_inferencedata=True,
            )

        # Point estimate: posterior mean
        beta_samples = self._trace.posterior["beta"].values  # (chains, draws, p)
        self._coef = beta_samples.reshape(-1, p).mean(axis=0)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self._coef

    @property
    def coef_(self) -> np.ndarray:
        return self._coef

    # -- Posterior intervals -----------------------------------------------

    @property
    def supports_intervals(self) -> bool:
        return True

    def posterior_intervals(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (lower, upper) HDI for each coefficient."""
        if self._trace is None:
            return None
        hdi = az.hdi(self._trace, var_names=["beta"], hdi_prob=1.0 - alpha)
        lower = hdi["beta"].values[:, 0]
        upper = hdi["beta"].values[:, 1]
        return lower, upper
