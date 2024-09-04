from abc import ABC, abstractmethod
from enum import Enum
from typing import override


import torch


class LossMethodsAbstract(torch.nn.Module, ABC):
    def __init__(self):
        super(LossMethodsAbstract, self).__init__()

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the Modified Huber Loss between `y_pred` and `y_true`.

        Parameters
        ----------
        y_pred : Tensor
            The predicted values.
        y_true : Tensor
            The ground truth values.

        Returns
        -------
        Tensor
            The mean loss value.
        """


class ModifiedHuberLoss(LossMethodsAbstract):
    def __init__(self, delta: float = 1.0, factor: float = 1.5) -> None:
        """
        Modified Huber Loss function with an additional factor for scaling the loss based on absolute error.

        Parameters
        ----------
        delta : float
            Threshold at which the loss function transitions from quadratic to linear. Default is 1.0.
        factor : float
            Factor by which the loss is scaled based on the absolute error. Default is 1.5.
        """

        super(ModifiedHuberLoss, self).__init__()
        self.delta = delta
        self.factor = factor

    @override
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # TODO : Test this method
        error = y_true - y_pred
        abs_error = torch.abs(error)
        delta_tensor = torch.tensor(self.delta, dtype=abs_error.dtype, device=abs_error.device)
        quadratic = torch.min(abs_error, delta_tensor)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + delta_tensor * linear
        return torch.mean(loss * (1 + self.factor * abs_error))


class LogCoshLoss(LossMethodsAbstract):
    def __init__(self, factor: float = 1.5) -> None:
        """
        Log-Cosh Loss function with an optional scaling factor.

        Parameters
        ----------
        factor : float
            Factor by which the loss is scaled based on the absolute error. Default is 1.5.
        """

        super(LogCoshLoss, self).__init__()
        self.factor = factor

    @override
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # TODO : Test this method
        error = y_true - y_pred
        logcosh = torch.log(torch.cosh(error))
        return torch.mean(logcosh * (1 + self.factor * torch.abs(error)))


class ExponentialLoss(LossMethodsAbstract):
    def __init__(self, alpha=0.5):
        """
        Exponential Loss function with a scaling parameter.

        Parameters
        ----------
        alpha : float
            Scaling factor for the exponential term. Default is 0.5.
        """
        super(ExponentialLoss, self).__init__()
        self.alpha = alpha

    @override
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # TODO : Test this method
        error = torch.abs(y_true - y_pred)
        loss = torch.exp(self.alpha * error) - 1
        return torch.mean(loss)


class LossMethodConstructors(Enum):
    """
    Enum class for the loss functions used in the neural network model.
    """

    MODIFIED_HUBER = ModifiedHuberLoss
    LOG_COSH = LogCoshLoss
    EXPONENTIAL = ExponentialLoss

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
