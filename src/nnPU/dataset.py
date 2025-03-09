# %%
import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
from transformers import AutoImageProcessor, SwiftFormerModel


class BinaryTargetTransformer:
    def __init__(
        self,
        included_classes: list[int],
        positive_classes: list[int],
        POSITIVE_CLASS=1,
        NEGATIVE_CLASS=-1,
    ) -> None:
        self._included_classes = included_classes
        self._positive_classes = positive_classes
        self._POSITIVE_CLASS = POSITIVE_CLASS
        self._NEGATIVE_CLASS = NEGATIVE_CLASS

    def transform(self, X, y):
        X = X[torch.isin(y, torch.tensor(self._included_classes))]
        y = y[torch.isin(y, torch.tensor(self._included_classes))]

        y = torch.where(
            torch.isin(y, torch.tensor(self._positive_classes)),
            self._POSITIVE_CLASS,
            self._NEGATIVE_CLASS,
        )
        return X, y


class PULabeler:
    def __init__(
        self, label_frequency: float, POSITIVE_LABEL=1, NEGATIVE_LABEL=-1
    ) -> None:
        self._label_frequency = label_frequency
        self._POSITIVE_LABEL = POSITIVE_LABEL
        self._NEGATIVE_LABEL = NEGATIVE_LABEL

    @property
    def label_frequency(self) -> float:
        return self._label_frequency

    @property
    def prior(self) -> float:
        raise NotImplementedError()

    def relabel(self, X, y):
        raise NotImplementedError()


class SCAR_SS_Labeler(PULabeler):
    def __init__(
        self, label_frequency: float, POSITIVE_LABEL=1, NEGATIVE_LABEL=-1
    ) -> None:
        super().__init__(
            label_frequency=label_frequency,
            POSITIVE_LABEL=POSITIVE_LABEL,
            NEGATIVE_LABEL=NEGATIVE_LABEL,
        )

    def relabel(self, X, y):
        self._prior = torch.mean((y == 1).float())

        labeling_condition = torch.rand_like(y, dtype=float) < self._label_frequency
        s = self.scar_targets = torch.where(
            y == 1,
            torch.where(
                labeling_condition,
                1,
                self._NEGATIVE_LABEL,
            ),
            self._NEGATIVE_LABEL,
        )

        return X, y, s

    @property
    def prior(self) -> float:
        return self._prior


class SCAR_CC_Labeler(PULabeler):
    def __init__(
        self, label_frequency: float, POSITIVE_LABEL=1, NEGATIVE_LABEL=-1
    ) -> None:
        super().__init__(
            label_frequency=label_frequency,
            POSITIVE_LABEL=POSITIVE_LABEL,
            NEGATIVE_LABEL=NEGATIVE_LABEL,
        )

    def relabel(self, X, y):
        self._prior = torch.mean((y == 1).float())

        n = len(y)
        c = self._label_frequency
        A = 1 / (1 - c + c * self._prior)

        P_samples_num = int(np.ceil(A * c * (self._prior * n)))
        U_samples_num = int(np.ceil(A * (1 - c) * n))

        if c != 0:
            positive_idx = torch.where(y == 1)[0]
            selected_positive_idx = torch.multinomial(
                torch.ones_like(positive_idx, dtype=torch.float32),
                P_samples_num,
                replacement=True,
            )
            positive_labeled_idx = positive_idx[selected_positive_idx]
        else:
            positive_labeled_idx = torch.tensor([], dtype=torch.int64)

        unlabeled_idx = torch.multinomial(
            torch.ones_like(y, dtype=torch.float32), U_samples_num, replacement=True
        )

        X = torch.cat(
            [
                X[positive_labeled_idx],
                X[unlabeled_idx],
            ]
        )
        y = torch.cat(
            [
                y[positive_labeled_idx],
                y[unlabeled_idx],
            ]
        )
        s = torch.cat(
            [
                torch.ones_like(positive_labeled_idx),
                self._NEGATIVE_LABEL * torch.ones_like(unlabeled_idx),
            ]
        )

        return X, y, s

    @property
    def prior(self) -> float:
        return self._prior


class PUDatasetBase:
    train: bool
    target_transformer: BinaryTargetTransformer
    pu_labeler: PULabeler

    data: list
    targets: list
    pu_targets: list
    binary_targets: list

    def __len__(self):
        return len(self.pu_targets)

    def __getitem__(self, idx):
        input = self.data[idx]
        target = self.binary_targets[idx]
        label = self.pu_targets[idx]
        return input, target, label

    def _convert_to_pu_data(self):
        assert self.target_transformer is not None
        assert self.pu_labeler is not None

        assert self.data is not None
        assert self.targets is not None

        self.data, self.binary_targets = self.target_transformer.transform(
            self.data, self.targets
        )
        self.data, self.binary_targets, self.pu_targets = self.pu_labeler.relabel(
            self.data, self.binary_targets
        )
        return self.data, self.binary_targets, self.pu_targets

    def get_prior(self):
        assert self.train is not None

        if self.train:
            return self.pu_labeler.prior
        else:
            return None


class DatasetSplitterMixin:
    def get_split_idx(
        self, dataset, split_type: Literal["train", "test"], random_seed, test_ratio=0.2
    ):
        assert (
            random_seed is not None
        ), "random_seed is necessary for a valid train / test split, please provide it"

        n_train = int((1 - test_ratio) * len(dataset))
        n_test = len(dataset) - n_train

        generator = torch.Generator().manual_seed(random_seed)
        train_idx, test_idx = random_split(
            range(len(dataset)), [n_train, n_test], generator=generator
        )

        if split_type == "train":
            return train_idx.indices
        return test_idx.indices


class SentenceTransformersDataset(Dataset):
    def __init__(
        self,
        root,
        dataset_hub_path,
        dataset_name,
        text_col="text",
        label_col="label",
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        dataset = load_dataset(
            dataset_hub_path, cache_dir=os.path.join(root, dataset_name)
        )
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.train = train
        if self.train:
            dataset = dataset["train"]
        else:
            dataset = dataset["test"]

        texts = dataset[text_col]
        self.data = torch.from_numpy(embedding_model.encode(texts))
        self.targets = torch.tensor(dataset[label_col])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            data = self.transform(data.numpy().reshape(1, *data.shape)).reshape(
                *data.shape
            )
        if self.target_transform is not None:
            target = self.transform(target)

        return data.numpy(), target


class TwentyNews_PU(PUDatasetBase, SentenceTransformersDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(20),
            positive_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="SetFit/20_newsgroups",
            dataset_name="20news",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class IMDB_PU(PUDatasetBase, SentenceTransformersDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="imdb",
            dataset_name="IMDB",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class HateSpeech_PU(PUDatasetBase, DatasetSplitterMixin, SentenceTransformersDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="hate_speech18",
            dataset_name="HateSpeech",
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.train = train
        if self.train:
            split_name = "train"
        else:
            split_name = "test"

        idx = self.get_split_idx(self.data, split_name, random_seed=random_seed)
        self.data = self.data[idx]
        self.targets = self.targets[idx]

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class SMSSpam_PU(PUDatasetBase, DatasetSplitterMixin, SentenceTransformersDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="sms_spam",
            dataset_name="SMSSpam",
            text_col="sms",
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.train = train
        if self.train:
            split_name = "train"
        else:
            split_name = "test"

        idx = self.get_split_idx(self.data, split_name, random_seed=random_seed)
        self.data = self.data[idx]
        self.targets = self.targets[idx]

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class PoemSentiment_PU(PUDatasetBase, SentenceTransformersDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(4),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="poem_sentiment",
            dataset_name="PoemSentiment",
            text_col="verse_text",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class TabularBenchmarkDataset(DatasetSplitterMixin, Dataset):
    def __init__(
        self,
        root,
        dataset_name,
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        dataset = load_dataset(
            "polinaeterna/tabular-benchmark",
            data_files=f"clf_num/{dataset_name}.csv",
            cache_dir=os.path.join(root, dataset_name),
        )
        dataset = dataset["train"]

        data = dataset.remove_columns(
            [dataset.column_names[0], dataset.column_names[-1]]
        )
        targets = dataset.select_columns(dataset.column_names[-1])

        X = pd.DataFrame(data).values
        y = pd.DataFrame(targets).values.reshape(-1)

        X = StandardScaler().fit_transform(X)

        self.train = train
        if self.train:
            split_name = "train"
        else:
            split_name = "test"

        idx = self.get_split_idx(dataset, split_name, random_seed=random_seed)
        X = X[idx]
        y = y[idx]

        self.data = torch.tensor(X).float()
        if y.dtype != object:
            self.targets = torch.tensor(y).int()
        else:
            self.targets = torch.from_numpy(
                np.where(y == np.unique(y)[0], 1, 0),
            )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            data = self.transform(data.numpy().reshape(1, *data.shape)).reshape(
                *data.shape
            )
        if self.target_transform is not None:
            target = self.transform(target)

        return data.numpy(), target


class TabularBenchmark_PU(PUDatasetBase, TabularBenchmarkDataset):
    def __init__(
        self,
        root,
        dataset_name,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmarkDataset.__init__(
            self,
            root=root,
            dataset_name=dataset_name,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            random_seed=random_seed,
        )

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class TBCredit_PU(TabularBenchmark_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmark_PU.__init__(
            self,
            root,
            "credit",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,  # ignored
            random_seed=random_seed,
        )


class TBCalifornia_PU(TabularBenchmark_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmark_PU.__init__(
            self,
            root,
            "california",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,  # ignored
            random_seed=random_seed,
        )


class TBWine_PU(TabularBenchmark_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmark_PU.__init__(
            self,
            root,
            "wine",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,  # ignored
            random_seed=random_seed,
        )


class TBElectricity_PU(TabularBenchmark_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmark_PU.__init__(
            self,
            root,
            "electricity",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,  # ignored
            random_seed=random_seed,
        )


class TBCovertype_PU(TabularBenchmark_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        transform=None,
        target_transform=None,
        download=True,  # ignored
        random_seed=None,
    ):
        TabularBenchmark_PU.__init__(
            self,
            root,
            "covertype",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,  # ignored
            random_seed=random_seed,
        )


class ImageEmbeddingDataset(DatasetSplitterMixin, PUDatasetBase):
    def __init__(
        self,
        root,
        dataset_hub_path,
        dataset_name,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer,
        image_col="img",
        label_col="label",
        train=True,
        download=True,  # ignored
        random_seed=None,
        dataset_hub_subset=None,
        image_preprocessing_fun=None,
        manually_split_dataset=False,
    ):
        dataset = load_dataset(
            dataset_hub_path,
            dataset_hub_subset,
            cache_dir=os.path.join(root, dataset_name),
        )

        self.train = train
        if not manually_split_dataset:
            if self.train:
                dataset = dataset["train"]
            else:
                dataset = dataset["test"]
        else:
            dataset = dataset["train"]
            if self.train:
                split_name = "train"
            else:
                split_name = "test"
            idx = self.get_split_idx(dataset, split_name, random_seed=random_seed)
            dataset = dataset.select(idx)

        def transforms(examples):
            image_processor = AutoImageProcessor.from_pretrained(
                "MBZUAI/swiftformer-xs"
            )
            model = SwiftFormerModel.from_pretrained("MBZUAI/swiftformer-xs").cuda()

            embeddings = []
            for img in examples[image_col]:
                if image_preprocessing_fun:
                    img = image_preprocessing_fun(img)

                inputs = image_processor(img, return_tensors="pt")
                inputs["pixel_values"] = inputs["pixel_values"].cuda()

                with torch.no_grad():
                    outputs = model(**inputs)

                last_hidden_states = outputs.last_hidden_state
                embeddings.append(last_hidden_states.reshape(-1).cpu())
            examples["data"] = embeddings
            return examples

        dataset = dataset.map(transforms, remove_columns=[image_col], batched=True)
        self.data = torch.tensor(dataset["data"])
        self.targets = torch.tensor(dataset[label_col])
        if self.targets.dtype == torch.bool:
            self.targets = torch.where(self.targets == True, 1, 0)

        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler
        self._convert_to_pu_data()


class CIFAR_PU(ImageEmbeddingDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(10),
            positive_classes=[2, 3, 4, 5, 6, 7],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="cifar10",
            dataset_name="CIFAR",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            image_col="img",
            label_col="label",
            train=train,
            download=download,
            random_seed=random_seed,
        )


class DogFood_PU(ImageEmbeddingDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(3),
            positive_classes=[1],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="lewtun/dog_food",
            dataset_name="DogFood",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            image_col="image",
            label_col="label",
            train=train,
            download=download,
            random_seed=random_seed,
        )


class Snacks_PU(ImageEmbeddingDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(20),
            positive_classes=[0, 1, 4, 7, 12, 13, 17, 19],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="Matthijs/snacks",
            dataset_name="Snacks",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            image_col="image",
            label_col="label",
            train=train,
            download=download,
            random_seed=random_seed,
        )


class MNIST_PU(ImageEmbeddingDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(10),
            positive_classes=[1, 3, 5, 7, 9],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="mnist",
            dataset_name="MNIST",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            image_col="image",
            label_col="label",
            train=train,
            download=download,
            random_seed=random_seed,
            image_preprocessing_fun=lambda img: img.convert("RGB"),
        )


class FashionMNIST_PU(ImageEmbeddingDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(10),
            positive_classes=[0, 2, 3, 4, 6],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="fashion_mnist",
            dataset_name="Fashion MNIST",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            image_col="image",
            label_col="label",
            train=train,
            download=download,
            random_seed=random_seed,
            image_preprocessing_fun=lambda img: img.convert("RGB"),
        )


class ChestXRay_PU(ImageEmbeddingDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[1],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="keremberke/chest-xray-classification",
            dataset_name="Chest X-ray",
            dataset_hub_subset="full",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            image_col="image",
            label_col="labels",
            train=train,
            download=download,
            random_seed=random_seed,
        )


class Beans_PU(ImageEmbeddingDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(3),
            positive_classes=[2],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="beans",
            dataset_name="Beans",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            image_col="image",
            label_col="labels",
            train=train,
            download=download,
            random_seed=random_seed,
        )


class OxfordPets_PU(ImageEmbeddingDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(2),
            positive_classes=[0],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="pcuenq/oxford-pets",
            dataset_name="Oxford Pets",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            image_col="image",
            label_col="dog",
            train=True,  # no test split defined
            download=download,
            random_seed=random_seed,
            manually_split_dataset=True,
            image_preprocessing_fun=lambda img: img.convert("RGB"),
        )


class EuroSAT_PU(ImageEmbeddingDataset):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(10),
            positive_classes=[3, 4, 7],  # Highway, Industrial, Residential
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
    ):
        super().__init__(
            root=root,
            dataset_hub_path="Ryukijano/eurosat",
            dataset_name="EuroSAT",
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            image_col="image",
            label_col="label",
            train=train,
            download=download,
            random_seed=random_seed,
            manually_split_dataset=True,
        )


class SyntheticPUDataset(PUDatasetBase):

    N = None
    PI = None
    MEAN = None

    def __init__(
        self,
        root,
        pu_labeler: PULabeler = None,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=[1, -1], positive_classes=[1]
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
    ) -> None:

        assert self.N is not None
        assert self.PI is not None
        assert self.MEAN is not None

        self.root = root
        self.train = train
        self.download = download
        self.random_seed = random_seed
        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler

        self.data = torch.cat(
            [
                torch.normal(0, 1, (int(self.PI * self.N), 10)),
                torch.normal(self.MEAN, 1, (int((1 - self.PI) * self.N), 10)),
            ]
        )
        self.targets = torch.cat(
            [
                torch.ones(int(self.PI * self.N)),
                -1 * torch.ones(int((1 - self.PI) * self.N)),
            ]
        )

        self._convert_to_pu_data()
