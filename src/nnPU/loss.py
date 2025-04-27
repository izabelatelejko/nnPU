from abc import abstractmethod

import torch
from torch import nn


class _PULoss(nn.Module):
    def __init__(
        self,
        prior,
        loss=(lambda x: torch.sigmoid(-x)),
        gamma=1,
        beta=0,
        nnPU=False,
        single_sample=False,
    ):
        super().__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss  # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.single_sample = single_sample
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1.0)

        self.labeled_component_history = []
        self.whole_distribution_cc_component_history = []
        self.whole_distribution_ss_component_history = []
        self.pu_scar_correction_component_history = []
        self.calculated_loss_history = []

    @property
    def history(self):
        return {
            "Labeled component": self.labeled_component_history,
            "Whole distribution component CC": self.whole_distribution_cc_component_history,
            "Whole distribution component SS": self.whole_distribution_ss_component_history,
            "PU SCAR correction": self.pu_scar_correction_component_history,
            "Calculated loss": self.calculated_loss_history,
        }

    def forward(self, x, target, test=False):
        assert x.shape == target.shape
        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        if x.is_cuda:
            self.min_count = self.min_count.cuda()
            self.prior = self.prior.cuda()
        n_positive, n_unlabeled = torch.max(
            self.min_count, torch.sum(positive)
        ), torch.max(self.min_count, torch.sum(unlabeled))
        n = n_positive + n_unlabeled

        y_positive = self.loss_func(x)
        y_unlabeled = self.loss_func(-x)

        positive_risk = self.prior * torch.sum(positive * y_positive) / n_positive
        negative_risk_c1_cc = torch.sum(unlabeled * y_unlabeled) / n_unlabeled
        negative_risk_c1_ss = torch.sum(y_unlabeled) / n
        negative_risk_c2 = self.prior * torch.sum(positive * y_unlabeled) / n_positive

        if not self.single_sample:
            negative_risk = negative_risk_c1_cc - negative_risk_c2
        else:
            negative_risk = negative_risk_c1_ss - negative_risk_c2

        if self.nnPU and negative_risk < -self.beta:
            loss = -self.gamma * negative_risk
        else:
            loss = positive_risk + negative_risk

        self._save_history(
            positive_risk,
            negative_risk_c1_cc,
            negative_risk_c1_ss,
            negative_risk_c2,
            loss,
        )

        return loss

    def _save_history(
        self,
        positive_risk,
        negative_risk_c1_cc,
        negative_risk_c1_ss,
        negative_risk_c2,
        loss,
    ):
        self.labeled_component_history.append(positive_risk.cpu().item())
        self.whole_distribution_cc_component_history.append(
            negative_risk_c1_cc.cpu().item()
        )
        self.whole_distribution_ss_component_history.append(
            negative_risk_c1_ss.cpu().item()
        )
        self.pu_scar_correction_component_history.append(negative_risk_c2.cpu().item())
        self.calculated_loss_history.append(loss.cpu().item())

    @property
    @abstractmethod
    def name():
        raise NotImplementedError("Implement in subclasses")


class nnPUccLoss(_PULoss):
    name = "nnPUcc"

    def __init__(
        self,
        prior,
        loss=lambda x: torch.sigmoid(-x),
        gamma=1,
        beta=0,
    ):
        super().__init__(prior, loss, gamma, beta, nnPU=True, single_sample=False)


class nnPUssLoss(_PULoss):
    name = "nnPUss"

    def __init__(
        self,
        prior,
        loss=lambda x: torch.sigmoid(-x),
        gamma=1,
        beta=0,
    ):
        super().__init__(prior, loss, gamma, beta, nnPU=True, single_sample=True)


class uPUccLoss(_PULoss):
    name = "uPUcc"

    def __init__(
        self,
        prior,
        loss=lambda x: torch.sigmoid(-x),
        gamma=1,
        beta=0,
    ):
        super().__init__(prior, loss, gamma, beta, nnPU=False, single_sample=False)


class uPUssLoss(_PULoss):
    name = "uPUss"

    def __init__(
        self,
        prior,
        loss=lambda x: torch.sigmoid(-x),
        gamma=1,
        beta=0,
    ):
        super().__init__(prior, loss, gamma, beta, nnPU=False, single_sample=True)


class DRPUccLoss(_PULoss):
    name = "DRPUcc"

    def __init__(
        self,
        prior,
        alpha,
        loss=(lambda x: torch.square(x - 1) / 2, lambda x: x - 1),
        gamma=1,
        beta=0,
    ):
        super().__init__(prior, loss, gamma, beta, nnPU=False, single_sample=False)
        self.alpha = alpha
        # Set the alpha to true train prior if it is known
        if self.prior is not None:
            self.alpha = self.prior
        self.f = loss[0]
        self.df = loss[1]
        self.f_dual = lambda x: x * self.df(x) - self.f(x)
        self.f_nn = lambda x: self.f_dual(x) - self.f_dual(0 * x)

    def forward(self, x, target, test=False):
        assert x.shape == target.shape
        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.bool), unlabeled.type(torch.bool)

        y_positive = x[positive]
        y_unlabeled = x[unlabeled]

        print(
            f"positive: {y_positive.shape}, unlabeled: {y_unlabeled.shape}, alpha: {self.alpha}"
        )

        E_pp = torch.mean(-self.df(y_positive) + self.alpha * self.f_nn(y_positive))
        E_pn = torch.mean(self.f_nn(y_positive))
        E_u = torch.mean(self.f_nn(y_unlabeled))
        E_n = E_u - self.alpha * E_pn
        # check if E_n is nan
        if torch.isnan(E_n):
            E_n = 0
        if torch.isnan(E_pp):
            E_pp = 0
        if torch.isnan(E_u):
            E_u = 0

        if E_n >= self.beta:
            loss = E_pp + max(0, E_n) + self.f_dual(0 * E_u)
        else:
            loss = -self.gamma * E_n

        print(f"E_pp: {E_pp}, E_pn: {E_pn}, E_u: {E_u}, E_n: {E_n}, loss: {loss}")
        return loss
