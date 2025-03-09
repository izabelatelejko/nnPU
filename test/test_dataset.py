import sys

import numpy as np
import pytest
import torch

sys.path.append(".")
from nnPU.dataset import (
    IMDB_PU,
    MNIST_PU,
    PUDatasetBase,
    PULabeler,
    SCAR_CC_Labeler,
    SCAR_SS_Labeler,
    TwentyNews_PU,
)

ss_dataset_configs = [
    (DatasetClass, SCAR_SS_Labeler, c)
    for DatasetClass in [
        MNIST_PU,
        TwentyNews_PU,
        IMDB_PU,
    ]
    for c in [0.1, 0.5, 0.9]
]
cc_dataset_configs = [
    (DatasetClass, SCAR_CC_Labeler, c)
    for DatasetClass in [
        MNIST_PU,
        TwentyNews_PU,
        IMDB_PU,
    ]
    for c in [0.1, 0.5, 0.9]
]


@pytest.mark.serial
@pytest.mark.parametrize("DatasetClass, LabelerClass, c", ss_dataset_configs)
def test_ss_prior(
    DatasetClass: type[PUDatasetBase], LabelerClass: type[PULabeler], c: float
):
    EPS = 0.05
    dataset = DatasetClass("data", LabelerClass(label_frequency=c), download=True)

    assert (
        np.abs(
            dataset.get_prior()
            - (dataset.binary_targets == 1).sum() / len(dataset.binary_targets)
        )
        < EPS
    )


@pytest.mark.serial
@pytest.mark.parametrize("DatasetClass, LabelerClass, c", cc_dataset_configs)
def test_cc_prior(
    DatasetClass: type[PUDatasetBase], LabelerClass: type[PULabeler], c: float
):
    EPS = 0.05
    dataset = DatasetClass("data", LabelerClass(label_frequency=c), download=True)

    assert (
        np.abs(
            dataset.get_prior()
            - ((dataset.binary_targets == 1) & (dataset.pu_targets == -1)).sum()
            / (dataset.pu_targets == -1).sum()
        )
        < EPS
    )


@pytest.mark.serial
@pytest.mark.parametrize("DatasetClass, LabelerClass, c", ss_dataset_configs)
def test_ss_proportions(
    DatasetClass: type[PUDatasetBase], LabelerClass: type[PULabeler], c: float
):
    EPS = 0.05
    dataset = DatasetClass("data", LabelerClass(label_frequency=c), download=True)

    empirical_c = torch.sum(dataset.pu_targets == 1) / torch.sum(
        dataset.binary_targets == 1
    )
    assert np.abs(empirical_c - c) < EPS, "Invalid label frequency in dataset"

    empirical_pi_l = torch.sum(dataset.pu_targets == 1) / len(dataset.pu_targets)
    assert (
        np.abs(empirical_pi_l - c * dataset.get_prior()) < EPS
    ), "Invalid labeled proportion in dataset"


@pytest.mark.serial
@pytest.mark.parametrize("DatasetClass, LabelerClass, c", cc_dataset_configs)
def test_cc_proportions(
    DatasetClass: type[PUDatasetBase], LabelerClass: type[PULabeler], c: float
):
    EPS = 0.05
    dataset = DatasetClass("data", LabelerClass(label_frequency=c), download=True)

    n = len(dataset.targets)
    assert np.abs(len(dataset) - n) < EPS * n, "Invalid length of dataset"

    empirical_c = torch.sum(dataset.pu_targets == 1) / torch.sum(
        dataset.binary_targets == 1
    )
    assert np.abs(empirical_c - c) < EPS, "Invalid label frequency in dataset"
