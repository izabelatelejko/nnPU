# %%
from nnPU.dataset_configs import DatasetConfigs
from nnPU.experiment_config import ExperimentConfig
from nnPU.loss import nnPUccLoss, nnPUssLoss, uPUccLoss, uPUssLoss
from nnPU.run_experiment import Experiment

if __name__ == "__main__":
    for exp_number in range(0, 10):
        for label_frequency in [
            0.9,
            0.7,
            0.5,
            0.3,
            0.1,
        ]:
            for dataset_config in [
                DatasetConfigs.CIFAR_CC,
                DatasetConfigs.CIFAR_SS,
                DatasetConfigs.MNIST_CC,
                DatasetConfigs.MNIST_SS,
                DatasetConfigs.FashionMNIST_CC,
                DatasetConfigs.FashionMNIST_SS,
                DatasetConfigs.EuroSAT_CC,
                DatasetConfigs.EuroSAT_SS,
                DatasetConfigs.ChestXRay_CC,
                DatasetConfigs.ChestXRay_SS,
                DatasetConfigs.Snacks_CC,
                DatasetConfigs.Snacks_SS,
                DatasetConfigs.DogFood_CC,
                DatasetConfigs.DogFood_SS,
                DatasetConfigs.Beans_CC,
                DatasetConfigs.Beans_SS,
                DatasetConfigs.OxfordPets_CC,
                DatasetConfigs.OxfordPets_SS,
                # //
                DatasetConfigs.TwentyNews_CC,
                DatasetConfigs.TwentyNews_SS,
                DatasetConfigs.IMDB_CC,
                DatasetConfigs.IMDB_SS,
                DatasetConfigs.HateSpeech_CC,
                DatasetConfigs.HateSpeech_SS,
                DatasetConfigs.SMSSpam_CC,
                DatasetConfigs.SMSSpam_SS,
                DatasetConfigs.PoemSentiment_CC,
                DatasetConfigs.PoemSentiment_SS,
                # //
                DatasetConfigs.TB_Credit_CC,
                DatasetConfigs.TB_Credit_SS,
                DatasetConfigs.TB_California_CC,
                DatasetConfigs.TB_California_SS,
                DatasetConfigs.TB_Wine_CC,
                DatasetConfigs.TB_Wine_SS,
                DatasetConfigs.TB_Electricity_CC,
                DatasetConfigs.TB_Electricity_SS,
            ]:
                for PULoss in [
                    nnPUssLoss,
                    nnPUccLoss,
                ]:
                    experiment_config = ExperimentConfig(
                        PULoss=PULoss,
                        dataset_config=dataset_config,
                        label_frequency=label_frequency,
                        exp_number=exp_number,
                    )
                    print(f"Starting: {experiment_config}...")

                    experiment = Experiment(experiment_config)
                    experiment.run()

                    print(f"Finished: {experiment_config}")

# %%
