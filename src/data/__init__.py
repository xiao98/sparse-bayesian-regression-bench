"""Data generation module: synthetic and real datasets for sparse regression benchmarks."""

from typing import Dict, Type
from src.data.base_dataset import BaseDataset
from src.data.synthetic import (
    IndependentGaussianDataset,
    BlockCorrelatedDataset,
    ToeplitzCorrelatedDataset,
)
from src.data.real_dataset import DiabetesDataset

dataset_name_to_class: Dict[str, Type[BaseDataset]] = {
    "Independent Gaussian": IndependentGaussianDataset,
    "Block Correlated": BlockCorrelatedDataset,
    "Toeplitz Correlated": ToeplitzCorrelatedDataset,
    "Diabetes": DiabetesDataset,
}
