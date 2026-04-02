"""Lasso regressor (L1 penalty — corresponds to Laplacian prior)."""

import numpy as np
from sklearn.linear_model import LassoCV

from src.models.base_model import BaseRegressor


class LassoRegressor(BaseRegressor):
    """Lasso with cross-validated regularization strength."""

    def __init__(self, config) -> None:
        super().__init__(config)
        n_alphas = int(config.get("n_alphas", 100))
        cv = int(config.get("cv", 5))
        max_iter = int(config.get("max_iter", 10000))
        self._model = LassoCV(
            n_alphas=n_alphas,
            cv=cv,
            max_iter=max_iter,
            fit_intercept=False,
            random_state=int(config.get("seed", 42)),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self._model.coef_
