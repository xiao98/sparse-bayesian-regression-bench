"""Model module: classical and Bayesian regressors for sparse regression."""

from typing import Dict, Type
from src.models.base_model import BaseRegressor
from src.models.ols import OLSRegressor
from src.models.lasso import LassoRegressor
from src.models.ridge import RidgeRegressor
from src.models.elastic_net import ElasticNetRegressor
from src.models.horseshoe import HorseshoeRegressor
from src.models.spike_slab import SpikeSlabRegressor

model_name_to_class: Dict[str, Type[BaseRegressor]] = {
    "OLS": OLSRegressor,
    "Lasso": LassoRegressor,
    "Ridge": RidgeRegressor,
    "Elastic Net": ElasticNetRegressor,
    "Horseshoe": HorseshoeRegressor,
    "Spike and Slab": SpikeSlabRegressor,
}
