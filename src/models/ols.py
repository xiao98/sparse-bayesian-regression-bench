"""Ordinary Least Squares regressor (baseline)."""

import numpy as np
from sklearn.linear_model import LinearRegression

from src.models.base_model import BaseRegressor


class OLSRegressor(BaseRegressor):
    """Plain OLS via sklearn — serves as the non-regularised baseline."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._model = LinearRegression(fit_intercept=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self._model.coef_
