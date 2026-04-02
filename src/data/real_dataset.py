"""Real dataset wrapper: sklearn Diabetes dataset.

Provides a compact real-world experiment to complement synthetic benchmarks.
Since true beta is unknown for real data, has_true_beta returns False.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.base_dataset import BaseDataset


class DiabetesDataset(BaseDataset):
    """Wrapper around sklearn's Diabetes dataset (442 samples, 10 features).

    Features are standardised; train/test split is controlled by seed.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        cfg = config
        test_size = float(cfg.get("test_size", 0.2))
        seed = int(cfg.get("seed", 42))

        data = load_diabetes()
        X, y = data.data, data.target  # type: ignore[attr-defined]

        # Standardise
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = (y - y.mean()) / y.std()

        self._X_train, self._X_test, self._y_train, self._y_test = (
            train_test_split(X, y, test_size=test_size, random_state=seed)
        )
        self._beta = np.zeros(X.shape[1])  # unknown ground truth

    # -- interface ---------------------------------------------------------

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

    @property
    def has_true_beta(self) -> bool:
        return False
