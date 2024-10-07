import json
import os
import time

import numpy as np
import pandas as pd
import pkbar
import torch
from experiment_config import ExperimentConfig
from metric_values import MetricValues
from model import PUModel
from sklearn import metrics
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import SyntheticPUDataset


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

        train_set = data["train"]
        print(data["train"])
        print(data["test"])
        self.prior = train_set.get_prior()
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.experiment_config.dataset_config.train_batch_size,
            shuffle=True,
        )
        self.n_inputs = len(next(iter(train_set))[0].reshape(-1))

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

        new_data_loader = DataLoader(
            new_data,
            batch_size=self.experiment_config.dataset_config.eval_batch_size,
            shuffle=False,
        )

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

        kbar.add(
            1,
            values=[
                ("new_data_loss", test_loss),
                ("new_data_accuracy", 100.0 * correct / len(self.test_loader.dataset)),
                ("pos_fraction", float(num_pos) / len(self.test_loader.dataset)),
            ],
        )

        targets = torch.cat(targets).detach().cpu().numpy()
        preds = torch.cat(preds).detach().cpu().numpy()

        metric_values = self._calculate_metrics(targets, preds)
        metric_values.loss = test_loss
        print("\n")
        print(metric_values)
        self.new_data_metrics = metric_values
        return metric_values

    def test_on_new_data_with_new_pi(self, new_data, pi, est_new_pi):
        self.model.eval()
        test_loss = 0
        correct = 0
        num_pos = 0

        test_points = []
        targets = []
        preds = []

        new_data_loader = DataLoader(
            new_data,
            batch_size=self.experiment_config.dataset_config.eval_batch_size,
            shuffle=False,
        )

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
                sigmoid_output = torch.sigmoid(output)

                test_loss += test_loss_func(
                    output.view(-1), target.type(torch.float)
                ).item()  # sum up batch loss

                tres = (pi / (1 - pi)) * ((1 - est_new_pi) / est_new_pi)
                print("tres: ", tres)

                pred = torch.where(
                    sigmoid_output / (1 - sigmoid_output) < tres,
                    torch.tensor(-1, device=self.device),
                    torch.tensor(1, device=self.device),
                )
                num_pos += torch.sum(pred == 1)
                correct += pred.eq(target.view_as(pred)).sum().item()

                test_points.append(data)
                targets.append(target)
                preds.append(pred)

        test_loss /= len(new_data_loader)

        kbar.add(
            1,
            values=[
                ("new_data_loss", test_loss),
                ("new_data_accuracy", 100.0 * correct / len(self.test_loader.dataset)),
                ("pos_fraction", float(num_pos) / len(self.test_loader.dataset)),
            ],
        )

        targets = torch.cat(targets).detach().cpu().numpy()
        preds = torch.cat(preds).detach().cpu().numpy()

        metric_values = self._calculate_metrics(targets, preds)
        metric_values.loss = test_loss
        print("\n")
        print(metric_values)
        self.new_data_metrics_with_new_pi = metric_values

        metric_values.n = new_data.N
        metric_values.pi = pi
        metric_values.new_pi = new_data.PI
        metric_values.est_new_pi = est_new_pi
        metric_values.treshold = tres
        metric_values.new_data_shifted_acc = metric_values.accuracy
        metric_values.new_data_shifted_precision = metric_values.precision
        metric_values.new_data_shifted_recall = metric_values.recall
        metric_values.new_data_shifted_f1 = metric_values.f1
        metric_values.new_data_shifted_auc = metric_values.auc

        metric_values.new_data_accuracy = self.new_data_metrics.accuracy
        metric_values.new_data_precision = self.new_data_metrics.precision
        metric_values.new_data_recall = self.new_data_metrics.recall
        metric_values.new_data_f1 = self.new_data_metrics.f1
        metric_values.new_data_auc = self.new_data_metrics.auc

        if isinstance(new_data, SyntheticPUDataset):
            metric_values.mean = new_data.MEAN
            metric_values.type = "synthetic"

        with open(self.experiment_config.new_data_metrics_file, "w") as f:
            json.dump(metric_values, f, cls=DictJsonEncoder)

        return metric_values
