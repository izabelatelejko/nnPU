from nnPU.dataset import (
    CIFAR_PU,
    IMDB_PU,
    MNIST_PU,
    Beans_PU,
    ChestXRay_PU,
    DogFood_PU,
    EuroSAT_PU,
    FashionMNIST_PU,
    HateSpeech_PU,
    OxfordPets_PU,
    PoemSentiment_PU,
    PUDatasetBase,
    PULabeler,
    SCAR_CC_Labeler,
    SCAR_SS_Labeler,
    SMSSpam_PU,
    Snacks_PU,
    TBCalifornia_PU,
    TBCredit_PU,
    TBElectricity_PU,
    TBWine_PU,
    TwentyNews_PU,
)
from torchvision import transforms


class DatasetConfig:
    def __init__(
        self,
        name: str,
        DatasetClass: type[PUDatasetBase],
        PULabelerClass: type[PULabeler],
        num_epochs=50,
        learning_rate=1e-5,
        train_batch_size=512,
        eval_batch_size=128,
    ):
        self.name = name
        self.DatasetClass = DatasetClass
        self.PULabelerClass = PULabelerClass
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size


class DatasetConfigs:
    MNIST_CC = DatasetConfig(
        "[IMG] MNIST CC",
        DatasetClass=MNIST_PU,
        PULabelerClass=SCAR_CC_Labeler,
    )
    MNIST_SS = DatasetConfig(
        "[IMG] MNIST SS",
        DatasetClass=MNIST_PU,
        PULabelerClass=SCAR_SS_Labeler,
    )

    FashionMNIST_CC = DatasetConfig(
        "[IMG] Fashion MNIST CC",
        DatasetClass=FashionMNIST_PU,
        PULabelerClass=SCAR_CC_Labeler,
    )
    FashionMNIST_SS = DatasetConfig(
        "[IMG] FashionMNIST SS",
        DatasetClass=FashionMNIST_PU,
        PULabelerClass=SCAR_SS_Labeler,
    )

    DogFood_CC = DatasetConfig(
        "[IMG] DogFood CC",
        DatasetClass=DogFood_PU,
        PULabelerClass=SCAR_CC_Labeler,
    )
    DogFood_SS = DatasetConfig(
        "[IMG] DogFood SS",
        DatasetClass=DogFood_PU,
        PULabelerClass=SCAR_SS_Labeler,
    )

    CIFAR_CC = DatasetConfig(
        "[IMG] CIFAR CC",
        DatasetClass=CIFAR_PU,
        PULabelerClass=SCAR_CC_Labeler,
    )
    CIFAR_SS = DatasetConfig(
        "[IMG] CIFAR SS",
        DatasetClass=CIFAR_PU,
        PULabelerClass=SCAR_SS_Labeler,
    )

    Snacks_CC = DatasetConfig(
        "[IMG] Snacks CC",
        DatasetClass=Snacks_PU,
        PULabelerClass=SCAR_CC_Labeler,
    )
    Snacks_SS = DatasetConfig(
        "[IMG] Snacks SS",
        DatasetClass=Snacks_PU,
        PULabelerClass=SCAR_SS_Labeler,
    )

    ChestXRay_CC = DatasetConfig(
        "[IMG] Chest X-ray CC",
        DatasetClass=ChestXRay_PU,
        PULabelerClass=SCAR_CC_Labeler,
    )
    ChestXRay_SS = DatasetConfig(
        "[IMG] Chest X-ray SS",
        DatasetClass=ChestXRay_PU,
        PULabelerClass=SCAR_SS_Labeler,
    )

    Beans_CC = DatasetConfig(
        "[IMG] Beans CC",
        DatasetClass=Beans_PU,
        PULabelerClass=SCAR_CC_Labeler,
    )
    Beans_SS = DatasetConfig(
        "[IMG] Beans SS",
        DatasetClass=Beans_PU,
        train_batch_size=64,
        PULabelerClass=SCAR_SS_Labeler,
    )

    OxfordPets_CC = DatasetConfig(
        "[IMG] Oxford Pets CC",
        DatasetClass=OxfordPets_PU,
        PULabelerClass=SCAR_CC_Labeler,
    )
    OxfordPets_SS = DatasetConfig(
        "[IMG] Oxford Pets SS",
        DatasetClass=OxfordPets_PU,
        PULabelerClass=SCAR_SS_Labeler,
    )

    EuroSAT_CC = DatasetConfig(
        "[IMG] EuroSAT CC",
        DatasetClass=EuroSAT_PU,
        PULabelerClass=SCAR_CC_Labeler,
    )
    EuroSAT_SS = DatasetConfig(
        "[IMG] EuroSAT SS",
        DatasetClass=EuroSAT_PU,
        PULabelerClass=SCAR_SS_Labeler,
    )

    TwentyNews_CC = DatasetConfig(
        "[TXT] 20News CC",
        DatasetClass=TwentyNews_PU,
        PULabelerClass=SCAR_CC_Labeler,
        learning_rate=5e-5,
        num_epochs=50,
    )
    TwentyNews_SS = DatasetConfig(
        "[TXT] 20News SS",
        DatasetClass=TwentyNews_PU,
        PULabelerClass=SCAR_SS_Labeler,
        learning_rate=5e-5,
        num_epochs=50,
    )

    IMDB_CC = DatasetConfig(
        "[TXT] IMDB CC",
        DatasetClass=IMDB_PU,
        PULabelerClass=SCAR_CC_Labeler,
    )
    IMDB_SS = DatasetConfig(
        "[TXT] IMDB SS",
        DatasetClass=IMDB_PU,
        PULabelerClass=SCAR_SS_Labeler,
    )

    HateSpeech_CC = DatasetConfig(
        "[TXT] HateSpeech CC",
        DatasetClass=HateSpeech_PU,
        PULabelerClass=SCAR_CC_Labeler,
        train_batch_size=64,
    )
    HateSpeech_SS = DatasetConfig(
        "[TXT] HateSpeech SS",
        DatasetClass=HateSpeech_PU,
        PULabelerClass=SCAR_SS_Labeler,
        train_batch_size=64,
    )

    SMSSpam_CC = DatasetConfig(
        "[TXT] SMSSpam CC",
        DatasetClass=SMSSpam_PU,
        PULabelerClass=SCAR_CC_Labeler,
        train_batch_size=64,
        num_epochs=100,
        learning_rate=1e-5,
    )
    SMSSpam_SS = DatasetConfig(
        "[TXT] SMSSpam SS",
        DatasetClass=SMSSpam_PU,
        PULabelerClass=SCAR_SS_Labeler,
        train_batch_size=64,
        num_epochs=100,
        learning_rate=1e-5,
    )

    PoemSentiment_CC = DatasetConfig(
        "[TXT] PoemSentiment CC",
        DatasetClass=PoemSentiment_PU,
        PULabelerClass=SCAR_CC_Labeler,
        train_batch_size=64,
        num_epochs=150,
    )
    PoemSentiment_SS = DatasetConfig(
        "[TXT] PoemSentiment SS",
        DatasetClass=PoemSentiment_PU,
        PULabelerClass=SCAR_SS_Labeler,
        train_batch_size=64,
        num_epochs=150,
    )

    TB_Credit_CC = DatasetConfig(
        "[TAB] Credit CC",
        DatasetClass=TBCredit_PU,
        PULabelerClass=SCAR_CC_Labeler,
        train_batch_size=128,
        num_epochs=100,
        learning_rate=2e-5,
    )
    TB_Credit_SS = DatasetConfig(
        "[TAB] Credit SS",
        DatasetClass=TBCredit_PU,
        PULabelerClass=SCAR_SS_Labeler,
        train_batch_size=128,
        num_epochs=100,
        learning_rate=2e-5,
    )

    TB_California_CC = DatasetConfig(
        "[TAB] California CC",
        DatasetClass=TBCalifornia_PU,
        PULabelerClass=SCAR_CC_Labeler,
        train_batch_size=128,
        num_epochs=100,
        learning_rate=2e-5,
    )
    TB_California_SS = DatasetConfig(
        "[TAB] California SS",
        DatasetClass=TBCalifornia_PU,
        PULabelerClass=SCAR_SS_Labeler,
        train_batch_size=128,
        num_epochs=100,
        learning_rate=2e-5,
    )

    TB_Wine_CC = DatasetConfig(
        "[TAB] Wine CC",
        DatasetClass=TBWine_PU,
        PULabelerClass=SCAR_CC_Labeler,
        train_batch_size=64,
        num_epochs=100,
    )
    TB_Wine_SS = DatasetConfig(
        "[TAB] Wine SS",
        DatasetClass=TBWine_PU,
        PULabelerClass=SCAR_SS_Labeler,
        train_batch_size=64,
        num_epochs=100,
    )

    TB_Electricity_CC = DatasetConfig(
        "[TAB] Electricity CC",
        DatasetClass=TBElectricity_PU,
        PULabelerClass=SCAR_CC_Labeler,
        num_epochs=200,
        learning_rate=5e-5,
    )
    TB_Electricity_SS = DatasetConfig(
        "[TAB] Electricity SS",
        DatasetClass=TBElectricity_PU,
        PULabelerClass=SCAR_SS_Labeler,
        num_epochs=200,
        learning_rate=5e-5,
    )
