"""Spike-and-Slab prior regressor via PyMC.

The spike-and-slab prior places:
    z_j        ~  Bernoulli(π)           (inclusion indicator)
    β_j | z_j  ~  z_j * N(0, σ_slab²)   (slab if included, spike ≈ 0 if not)
    π          ~  Beta(a, b)             (inclusion probability)

This uses a continuous relaxation via a mixture to avoid discrete sampling.
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


class SpikeSlabRegressor(BaseRegressor):
    """Spike-and-Slab prior fitted with NUTS via PyMC (continuous relaxation)."""

    def __init__(self, config) -> None:
        super().__init__(config)
        if not _HAS_PYMC:
            raise ImportError("PyMC is required for SpikeSlabRegressor. "
                              "Install via: pip install pymc arviz")
        self.n_samples = int(config.get("n_samples", 2000))
        self.n_tune = int(config.get("n_tune", 1000))
        self.n_chains = int(config.get("n_chains", 2))
        self.target_accept = float(config.get("target_accept", 0.95))
        self.slab_scale = float(config.get("slab_scale", 5.0))
        self.spike_scale = float(config.get("spike_scale", 0.01))
        self.inclusion_prob = float(config.get("inclusion_prob", 0.2))
        self.seed = int(config.get("seed", 42))
        self._coef = None
        self._trace = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p = X.shape

        with pm.Model() as model:
            # Inclusion probability (can learn from data)
            pi = pm.Beta("pi", alpha=1.0, beta=1.0 / self.inclusion_prob)

            # Continuous relaxation: mixture of narrow + wide Gaussians
            # β_j ~ π * N(0, slab²) + (1-π) * N(0, spike²)
            w = pm.math.stack([1.0 - pi, pi])
            comp_sds = [self.spike_scale, self.slab_scale]

            beta = pm.NormalMixture(
                "beta",
                w=w,
                mu=0.0,
                sigma=comp_sds,
                shape=p,
            )

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

        beta_samples = self._trace.posterior["beta"].values
        p_dim = beta_samples.shape[-1]
        self._coef = beta_samples.reshape(-1, p_dim).mean(axis=0)
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
        if self._trace is None:
            return None
        hdi = az.hdi(self._trace, var_names=["beta"], hdi_prob=1.0 - alpha)
        lower = hdi["beta"].values[:, 0]
        upper = hdi["beta"].values[:, 1]
        return lower, upper
