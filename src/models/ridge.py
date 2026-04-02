"""Ridge regressor (L2 penalty — corresponds to Gaussian prior)."""

import numpy as np
from sklearn.linear_model import RidgeCV

from src.models.base_model import BaseRegressor


class RidgeRegressor(BaseRegressor):
    """Ridge with cross-validated regularization strength."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._model = RidgeCV(
            alphas=np.logspace(-4, 4, 50),
            fit_intercept=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self._model.coef_
