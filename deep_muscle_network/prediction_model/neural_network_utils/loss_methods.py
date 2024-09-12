from abc import ABC, abstractmethod
from enum import Enum
from typing import override

import torch


class LossFunctionAbstract(torch.nn.Module, ABC):
    def __init__(self):
        super(LossFunctionAbstract, self).__init__()

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

    def serialize(self) -> dict:
        """
        Serialize the loss function.

        Returns
        -------
        dict
            The serialized loss function.
        """

        return {"loss_type": self.__class__.__name__, "parameters": self._serialize()}

    @abstractmethod
    def _serialize(self) -> dict:
        """
        Serialize the loss function. This method should return a dictionary that will be sent back when deserializing the
        loss function. The constructor should be able to use this dictionary to reconstruct the loss function.

        Returns
        -------
        dict
            The serialized loss function.
        """

    @classmethod
    def deserialize(cls, parameters: dict):
        """
        Deserialize the loss function.

        Parameters
        ----------
        parameters : dict
            The serialized loss function.

        Returns
        -------
        LossFunctionAbstract
            The deserialized loss function.
        """

        loss_class_constructor = globals()[parameters["loss_type"]]
        parameters = parameters["parameters"]
        return loss_class_constructor(**parameters)

    @abstractmethod
    def _deserialize(self, **parameters: dict):
        """
        Deserialize the loss function. This method should return a new instance of the loss function using the
        serialized dictionary.

        Parameters
        ----------
        serialized_loss_function : dict
            The serialized loss function.

        Returns
        -------
        LossFunctionAbstract
            The deserialized loss function.
        """


class LossFunctionModifiedHuber(LossFunctionAbstract):
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

        super(LossFunctionModifiedHuber, self).__init__()
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
        return torch.mean(loss * (1 + self.factor * abs_error), axis=1)

    @override
    def _serialize(self) -> dict:
        return {"delta": self.delta, "factor": self.factor}

    @classmethod
    def _deserialize(cls, **serialized_loss_function):
        return cls(**serialized_loss_function)


class LossFunctionLogCosh(LossFunctionAbstract):
    def __init__(self, factor: float = 1.5) -> None:
        """
        Log-Cosh Loss function with an optional scaling factor.

        Parameters
        ----------
        factor : float
            Factor by which the loss is scaled based on the absolute error. Default is 1.5.
        """

        super(LossFunctionLogCosh, self).__init__()
        self.factor = factor

    @override
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # TODO : Test this method
        error = y_true - y_pred
        logcosh = torch.log(torch.cosh(error))
        return torch.mean(logcosh * (1 + self.factor * torch.abs(error)), axis=1)

    @override
    def _serialize(self) -> dict:
        return {"factor": self.factor}

    @classmethod
    def _deserialize(cls, **parameters):
        return cls(**parameters)


class LossFunctionExponential(LossFunctionAbstract):
    def __init__(self, alpha=0.5):
        """
        Exponential Loss function with a scaling parameter.

        Parameters
        ----------
        alpha : float
            Scaling factor for the exponential term. Default is 0.5.
        """
        super(LossFunctionExponential, self).__init__()
        self.alpha = alpha

    @override
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # TODO : Test this method
        error = torch.abs(y_true - y_pred)
        loss = torch.exp(self.alpha * error) - 1
        return torch.mean(loss, axis=1)

    @override
    def _serialize(self) -> dict:
        return {"alpha": self.alpha}

    @classmethod
    def _deserialize(cls, **parameters):
        return cls(**parameters)


class LossFunctionConstructors(Enum):
    """
    Enum class for the loss functions used in the neural network model.
    """

    MODIFIED_HUBER = LossFunctionModifiedHuber
    LOG_COSH = LossFunctionLogCosh
    EXPONENTIAL = LossFunctionExponential

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
