# %%
import numpy as np
import pandas as pd
import torch

from nnPU.dataset_configs import DatasetConfigs

stats = []
for dataset_config in [
    DatasetConfigs.CIFAR_SS,
    DatasetConfigs.MNIST_SS,
    DatasetConfigs.FashionMNIST_SS,
    DatasetConfigs.EuroSAT_SS,
    DatasetConfigs.ChestXRay_SS,
    DatasetConfigs.Snacks_SS,
    DatasetConfigs.DogFood_SS,
    DatasetConfigs.Beans_SS,
    DatasetConfigs.OxfordPets_SS,
    # //
    DatasetConfigs.TwentyNews_SS,
    DatasetConfigs.IMDB_SS,
    DatasetConfigs.HateSpeech_SS,
    DatasetConfigs.SMSSpam_SS,
    DatasetConfigs.PoemSentiment_SS,
    # //
    DatasetConfigs.TB_Credit_SS,
    DatasetConfigs.TB_California_SS,
    DatasetConfigs.TB_Wine_SS,
    DatasetConfigs.TB_Electricity_SS,
]:
    dataset = dataset_config.DatasetClass(
        "data",
        dataset_config.PULabelerClass(label_frequency=0.5),
        train=True,
        download=True,
        random_seed=1,
    )

    stats.append(
        {
            "Dataset": dataset_config.name,
            "Samples": len(dataset),
            "Features": len(next(iter(dataset.base_dataset))["data"]),
            "$\pi$": np.round(
                torch.sum(dataset.binary_targets == 1).item()
                / len(dataset.binary_targets),
                2,
            ),
        }
    )

stats = pd.DataFrame.from_records(stats)
stats.to_latex("latex/dataset-stats.tex", index=False)

# %%
