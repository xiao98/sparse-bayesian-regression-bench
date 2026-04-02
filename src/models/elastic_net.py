"""Elastic Net regressor (L1 + L2 penalty)."""

import numpy as np
from sklearn.linear_model import ElasticNetCV

from src.models.base_model import BaseRegressor


class ElasticNetRegressor(BaseRegressor):
    """Elastic Net with cross-validated alpha and l1_ratio."""

    def __init__(self, config) -> None:
        super().__init__(config)
        cv = int(config.get("cv", 5))
        max_iter = int(config.get("max_iter", 10000))
        self._model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
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
