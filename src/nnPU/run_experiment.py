import json
import os
import time

import numpy as np
import pandas as pd
import pkbar
import torch
from nnPU.experiment_config import ExperimentConfig
from nnPU.metric_values import MetricValues
from nnPU.model import PUModel
from sklearn import metrics
from torch.optim import Adam
from torch.utils.data import DataLoader

from nnPUss.dataset import SyntheticPUDataset

from DRPU.algorithm import priorestimator as ratio_estimator
from DRPU.algorithm import PUsequence, to_ndarray


class DictJsonEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


class Experiment:
    def __init__(self, experiment_config: ExperimentConfig) -> None:
        self.experiment_config = experiment_config

        use_cuda = not self.experiment_config.force_cpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._prepare_data()

        self._set_seed()

        self.model = PUModel(self.n_inputs)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.experiment_config.dataset_config.learning_rate,
            weight_decay=0.005,
        )

        self.train_metrics = []
        self.test_metrics = []

        self.new_data_metrics = None
        self.new_data_metrics_with_new_pi = None

    def run(self):
        os.makedirs(self.experiment_config.output_dir, exist_ok=True)

        self._set_seed()

        loss_details_history = []
        self.model = self.model.to(self.device)
        training_start_time = time.perf_counter()
        for epoch in range(self.experiment_config.dataset_config.num_epochs):
            kbar = pkbar.Kbar(
                target=len(self.train_loader) + 1,
                epoch=epoch,
                num_epochs=self.experiment_config.dataset_config.num_epochs,
                width=8,
                always_stateful=False,
            )

            loss_details = self._train(epoch, kbar)
            loss_details["Epoch"] = epoch
            loss_details_history.append(loss_details)
            self._test(epoch, kbar)

        pd.DataFrame.from_records(loss_details_history).to_csv(
            os.path.join(self.experiment_config.output_dir, "loss-history.csv"),
            index=False,
        )

        self.training_time = time.perf_counter() - training_start_time

        kbar = pkbar.Kbar(
            target=1,
            epoch=epoch,
            num_epochs=self.experiment_config.dataset_config.num_epochs,
            width=8,
            always_stateful=False,
        )
        self._test(epoch, kbar, save_metrics=True)

    def _prepare_data(self):
        self._set_seed()

        data = {}
        for is_train_set in [True, False]:
            data["train" if is_train_set else "test"] = (
                self.experiment_config.dataset_config.DatasetClass(
                    self.experiment_config.data_dir,
                    self.experiment_config.dataset_config.PULabelerClass(
                        label_frequency=self.experiment_config.label_frequency
                    ),
                    train=is_train_set,
                    download=True,
                    random_seed=self.experiment_config.seed,
                )
            )

        self.train_set = data["train"]
        self.prior = self.train_set.get_prior()
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.experiment_config.dataset_config.train_batch_size,
            shuffle=True,
        )
        self.n_inputs = len(next(iter(self.train_set))[0].reshape(-1))

        test_set = data["test"]
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.experiment_config.dataset_config.eval_batch_size,
            shuffle=False,
        )

    def _train(self, epoch: int, kbar: pkbar.Kbar):
        self.model.train()
        tr_loss = 0

        targets = []
        preds = []
        loss_fct = self.experiment_config.PULoss(prior=self.prior)
        for batch_idx, (data, target, label) in enumerate(self.train_loader):
            data, label = data.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = loss_fct(output.view(-1), label.type(torch.float))
            tr_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            y_pred = torch.where(output < 0, -1, 1).to(self.device)
            acc = metrics.accuracy_score(
                target.cpu().numpy().reshape(-1), y_pred.cpu().numpy().reshape(-1)
            )
            kbar.update(batch_idx + 1, values=[("loss", loss), ("acc", acc)])

            targets.append(target.reshape(-1))
            preds.append(y_pred.reshape(-1))

        targets = torch.cat(targets).detach().cpu().numpy()
        preds = torch.cat(preds).detach().cpu().numpy()

        metric_values = self._calculate_metrics(targets, preds)
        metric_values.loss = tr_loss
        metric_values.epoch = epoch
        self.train_metrics.append(metric_values)

        history = loss_fct.history
        history_means = {k: np.mean(v) for k, v in history.items()}
        return history_means

    def _test(self, epoch: int, kbar: pkbar.Kbar, save_metrics: bool = False):
        """Testing"""
        self.model.eval()
        test_loss = 0
        correct = 0
        num_pos = 0

        test_points = []
        targets = []
        preds = []
        with torch.no_grad():
            test_loss_func = self.experiment_config.PULoss(prior=self.prior)
            for data, target, _ in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                test_loss += test_loss_func(
                    output.view(-1), target.type(torch.float)
                ).item()  # sum up batch loss
                sigmoid_output = torch.sigmoid(output)
                pred = torch.where(
                    sigmoid_output / (1 - sigmoid_output) < 1,
                    torch.tensor(-1, device=self.device),
                    torch.tensor(1, device=self.device),
                )
                num_pos += torch.sum(pred == 1)
                correct += pred.eq(target.view_as(pred)).sum().item()

                test_points.append(data)
                targets.append(target)
                preds.append(pred)

        test_loss /= len(self.test_loader)

        kbar.add(
            1,
            values=[
                ("test_loss", test_loss),
                ("test_accuracy", 100.0 * correct / len(self.test_loader.dataset)),
                ("pos_fraction", float(num_pos) / len(self.test_loader.dataset)),
            ],
        )

        targets = torch.cat(targets).detach().cpu().numpy()
        preds = torch.cat(preds).detach().cpu().numpy()

        metric_values = self._calculate_metrics(targets, preds)
        metric_values.loss = test_loss
        metric_values.epoch = epoch
        self.test_metrics.append(metric_values)

        if save_metrics:
            metric_values.time = self.training_time

            with open(self.experiment_config.metrics_file, "w") as f:
                json.dump(metric_values, f, cls=DictJsonEncoder)
            with open(self.experiment_config.train_metrics_per_epoch_file, "w") as f:
                json.dump(self.train_metrics, f, cls=DictJsonEncoder)
            with open(self.experiment_config.test_metrics_per_epoch_file, "w") as f:
                json.dump(self.test_metrics, f, cls=DictJsonEncoder)

        return test_loss

    def _calculate_metrics(self, targets, preds):
        y_true = np.where(targets == 1, 1, 0)
        y_pred = np.where(preds == 1, 1, 0)

        metric_values = MetricValues(
            model=self.experiment_config.PULoss.name,
            dataset=self.experiment_config.dataset_config.name,
            label_frequency=self.experiment_config.label_frequency,
            exp_number=self.experiment_config.exp_number,
            accuracy=metrics.accuracy_score(y_true, y_pred),
            precision=metrics.precision_score(y_true, y_pred),
            recall=metrics.recall_score(y_true, y_pred),
            f1=metrics.f1_score(y_true, y_pred),
            auc=metrics.roc_auc_score(y_true, y_pred),
        )

        return metric_values

    def _set_seed(self):
        torch.manual_seed(self.experiment_config.seed)
        np.random.seed(self.experiment_config.seed)

    def test_on_new_data(self, new_data):
        self.model.eval()
        test_loss = 0
        correct = 0
        num_pos = 0

        test_points = []
        targets = []
        preds = []
        outputs = []

        new_data_loader = DataLoader(
            new_data,
            batch_size=self.experiment_config.dataset_config.eval_batch_size,
            shuffle=False,
        )
        # self.new_data_loader = new_data_loader

        kbar = pkbar.Kbar(
            target=len(new_data_loader) + 1,
            epoch=0,
            num_epochs=1,
            width=8,
            always_stateful=False,
        )
        with torch.no_grad():
            test_loss_func = self.experiment_config.PULoss(prior=self.prior)
            for data, target, _ in new_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                outputs.append(output)

                test_loss += test_loss_func(
                    output.view(-1), target.type(torch.float)
                ).item()  # sum up batch loss
                pred = torch.where(
                    output < 0,
                    torch.tensor(-1, device=self.device),
                    torch.tensor(1, device=self.device),
                )
                num_pos += torch.sum(pred == 1)
                correct += pred.eq(target.view_as(pred)).sum().item()

                test_points.append(data)
                targets.append(target)
                preds.append(pred)

        test_loss /= len(new_data_loader)
        pos_fraction = float(num_pos) / len(new_data_loader.dataset)

        kbar.add(
            1,
            values=[
                ("new_data_loss", test_loss),
                ("new_data_accuracy", 100.0 * correct / len(self.test_loader.dataset)),
                ("pos_fraction", pos_fraction),
            ],
        )

        targets = torch.cat(targets).detach().cpu().numpy()
        preds = torch.cat(preds).detach().cpu().numpy()

        metric_values = self._calculate_metrics(targets, preds)
        metric_values.loss = test_loss
        metric_values.pos_fraction = pos_fraction
        print("\n")
        print(metric_values)
        self.new_data_metrics = metric_values
        return metric_values

    def test_on_new_data_with_new_pi(
        self,
        new_data,
        pi,
        km1_pi,
        km2_pi,
        true_new_pi,
    ):
        self.model.eval()
        test_loss = 0
        true_correct = 0
        km1_correct = 0
        km2_correct = 0
        ratio_correct = 0
        true_num_pos = 0
        km1_num_pos = 0
        km2_num_pos = 0
        ratio_num_pos = 0

        test_points = []
        targets = []
        true_preds = []
        km1_preds = []
        km2_preds = []
        ratio_preds = []
        outputs = []

        new_data_loader = DataLoader(
            new_data,
            batch_size=self.experiment_config.dataset_config.eval_batch_size,
            shuffle=False,
        )
        ratio_pi = self.estimate_ratio_prior(new_data_loader)

        kbar = pkbar.Kbar(
            target=len(new_data_loader) + 1,
            epoch=0,
            num_epochs=1,
            width=8,
            always_stateful=False,
        )
        print(f"prior: {self.prior}")

        with torch.no_grad():
            test_loss_func = self.experiment_config.PULoss(prior=self.prior)
            for data, target, _ in new_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                outputs.append(output)
                sigmoid_output = torch.sigmoid(output)

                test_loss += test_loss_func(
                    output.view(-1), target.type(torch.float)
                ).item()  # sum up batch loss

                true_tres = (pi / (1 - pi)) * ((1 - true_new_pi) / true_new_pi)
                km1_tres = (pi / (1 - pi)) * ((1 - km1_pi) / km1_pi)
                km2_tres = (pi / (1 - pi)) * ((1 - km2_pi) / km2_pi)
                ratio_tres = (pi / (1 - pi)) * ((1 - ratio_pi) / ratio_pi)

                true_pred = torch.where(
                    sigmoid_output / (1 - sigmoid_output) < true_tres,
                    torch.tensor(-1, device=self.device),
                    torch.tensor(1, device=self.device),
                )
                km1_pred = torch.where(
                    sigmoid_output / (1 - sigmoid_output) < km1_tres,
                    torch.tensor(-1, device=self.device),
                    torch.tensor(1, device=self.device),
                )
                km2_pred = torch.where(
                    sigmoid_output / (1 - sigmoid_output) < km2_tres,
                    torch.tensor(-1, device=self.device),
                    torch.tensor(1, device=self.device),
                )
                ratio_pred = torch.where(
                    sigmoid_output / (1 - sigmoid_output) < ratio_tres,
                    torch.tensor(-1, device=self.device),
                    torch.tensor(1, device=self.device),
                )

                true_num_pos += torch.sum(true_pred == 1)
                km1_num_pos += torch.sum(km1_pred == 1)
                km2_num_pos += torch.sum(km2_pred == 1)
                ratio_num_pos += torch.sum(ratio_pred == 1)

                true_correct += true_pred.eq(target.view_as(true_pred)).sum().item()
                km1_correct += km1_pred.eq(target.view_as(km1_pred)).sum().item()
                km2_correct += km2_pred.eq(target.view_as(km2_pred)).sum().item()
                ratio_correct += ratio_pred.eq(target.view_as(ratio_pred)).sum().item()

                test_points.append(data)
                targets.append(target)

                true_preds.append(true_pred)
                km1_preds.append(km1_pred)
                km2_preds.append(km2_pred)
                ratio_preds.append(ratio_pred)

        test_loss /= len(new_data_loader)
        true_pos_fraction = float(true_num_pos) / len(new_data_loader.dataset)
        km1_pos_fraction = float(km1_num_pos) / len(new_data_loader.dataset)
        km2_pos_fraction = float(km2_num_pos) / len(new_data_loader.dataset)
        ratio_pos_fraction = float(ratio_num_pos) / len(new_data_loader.dataset)

        kbar.add(
            1,
            values=[
                ("new_data_loss", test_loss),
                ("true_accuracy", 100.0 * true_correct / len(self.test_loader.dataset)),
                ("true_pos_fraction", true_pos_fraction),
                ("km1_accuracy", 100.0 * km1_correct / len(self.test_loader.dataset)),
                ("km1_pos_fraction", km1_pos_fraction),
                ("km2_accuracy", 100.0 * km2_correct / len(self.test_loader.dataset)),
                ("km2_pos_fraction", km2_pos_fraction),
                (
                    "ratio_accuracy",
                    100.0 * ratio_correct / len(self.test_loader.dataset),
                ),
                ("ratio_pos_fraction", ratio_pos_fraction),
            ],
        )

        targets = torch.cat(targets).detach().cpu().numpy()
        true_preds = torch.cat(true_preds).detach().cpu().numpy()
        km1_preds = torch.cat(km1_preds).detach().cpu().numpy()
        km2_preds = torch.cat(km2_preds).detach().cpu().numpy()
        ratio_preds = torch.cat(ratio_preds).detach().cpu().numpy()
        outputs = torch.cat(outputs).detach().cpu().numpy()

        true_metric_values = self._calculate_metrics(targets, true_preds)
        km1_metric_values = self._calculate_metrics(targets, km1_preds)
        km2_metric_values = self._calculate_metrics(targets, km2_preds)
        ratio_metric_values = self._calculate_metrics(targets, ratio_preds)

        # self.new_data_metrics_with_new_pi = metric_values

        metric_values = MetricValues(
            model=self.experiment_config.PULoss.name,
            dataset=self.experiment_config.dataset_config.name,
            label_frequency=self.experiment_config.label_frequency,
            exp_number=self.experiment_config.exp_number,
            loss=test_loss,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            auc=None,
        )

        metric_values.n = new_data.N
        metric_values.pi = pi
        metric_values.new_pi = new_data.PI

        metric_values.not_adjusted_accuracy = self.new_data_metrics.accuracy
        metric_values.not_adjusted_precision = self.new_data_metrics.precision
        metric_values.not_adjusted_recall = self.new_data_metrics.recall
        metric_values.not_adjusted_f1 = self.new_data_metrics.f1
        metric_values.not_adjusted_auc = self.new_data_metrics.auc
        metric_values.not_adjusted_pos_fraction = self.new_data_metrics.pos_fraction

        metric_values.true_pi_tres = true_tres
        metric_values.true_pi_accuracy = true_metric_values.accuracy
        metric_values.true_pi_precision = true_metric_values.precision
        metric_values.true_pi_recall = true_metric_values.recall
        metric_values.true_pi_f1 = true_metric_values.f1
        metric_values.true_pi_auc = true_metric_values.auc
        metric_values.true_pi_pos_fraction = true_pos_fraction

        metric_values.km1_pi = km1_pi
        metric_values.km1_tres = km1_tres
        metric_values.km1_accuracy = km1_metric_values.accuracy
        metric_values.km1_precision = km1_metric_values.precision
        metric_values.km1_recall = km1_metric_values.recall
        metric_values.km1_f1 = km1_metric_values.f1
        metric_values.km1_auc = km1_metric_values.auc
        metric_values.km1_pos_fraction = km1_pos_fraction

        metric_values.km2_pi = km2_pi
        metric_values.km2_tres = km2_tres
        metric_values.km2_accuracy = km2_metric_values.accuracy
        metric_values.km2_precision = km2_metric_values.precision
        metric_values.km2_recall = km2_metric_values.recall
        metric_values.km2_f1 = km2_metric_values.f1
        metric_values.km2_auc = km2_metric_values.auc
        metric_values.km2_pos_fraction = km2_pos_fraction

        metric_values.ratio_pi = ratio_pi
        metric_values.ratio_tres = ratio_tres
        metric_values.ratio_accuracy = ratio_metric_values.accuracy
        metric_values.ratio_precision = ratio_metric_values.precision
        metric_values.ratio_recall = ratio_metric_values.recall
        metric_values.ratio_f1 = ratio_metric_values.f1
        metric_values.ratio_auc = ratio_metric_values.auc
        metric_values.ratio_pos_fraction = ratio_pos_fraction

        if isinstance(new_data, SyntheticPUDataset):
            metric_values.mean = new_data.MEAN
            metric_values.type = "synthetic"

        with open(self.experiment_config.new_data_metrics_file, "w") as f:
            json.dump(metric_values, f, cls=DictJsonEncoder)

        print("\n")
        print(metric_values)

        return metric_values

    def estimate_ratio_prior(self, new_data_loader):
        self.model.eval()
        with torch.no_grad():
            preds_P, preds_U = [], []

            # positive from training set
            for data, target, _ in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                preds = self.model(data)  # .view(-1)
                preds = preds[target == 1]
                preds_P.append(to_ndarray(preds))  # to_ndarray(y))

            # unlabeled from new data
            for data, target, _ in new_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                preds = self.model(data)  # .view(-1)
                preds_U.append(to_ndarray(preds))  # to_ndarray(y))

            preds_P = np.concatenate(preds_P)
            preds_U = np.concatenate(preds_U)

            prior = ratio_estimator(
                np.concatenate([preds_P, preds_U]),
                PUsequence(len(preds_P), len(preds_U)),
            )

        return prior
