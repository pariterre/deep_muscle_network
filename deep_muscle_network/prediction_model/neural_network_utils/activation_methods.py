from abc import ABC, abstractmethod
from enum import Enum
from typing import override

import torch


class ActivationMethodAbstract(torch.nn.Module, ABC):
    def __init__(self):
        super(ActivationMethodAbstract, self).__init__()

    def serialize(self) -> dict:
        """
        Serialize the activation method.

        Returns
        -------
        dict
            The serialized activation method.
        """

        return {"loss_type": self.__class__.__name__, "parameters": self._serialize()}

    @abstractmethod
    def _serialize(self) -> dict:
        """
        Serialize the activation method. This method should return a dictionary that will be sent back when deserializing the
        activation method. The constructor should be able to use this dictionary to reconstruct the activation method.

        Returns
        -------
        dict
            The serialized activation method.
        """

    @classmethod
    def deserialize(cls, parameters: dict):
        """
        Deserialize the activation method.

        Parameters
        ----------
        serialized_loss_function : dict
            The serialized activation method.

        Returns
        -------
        LossFunctionAbstract
            The deserialized activation method.
        """

        loss_class_constructor = globals()[parameters["loss_type"]]
        parameters = parameters["parameters"]
        return loss_class_constructor(**parameters)

    @abstractmethod
    def _deserialize(self, **parameters: dict):
        """
        Deserialize the activation method. This method should return a new instance of the activation method using the
        serialized dictionary.

        Parameters
        ----------
        serialized_loss_function : dict
            The serialized activation method.

        Returns
        -------
        LossFunctionAbstract
            The deserialized activation method.
        """


class GeLU(torch.nn.GELU, ActivationMethodAbstract):
    def __init__(self) -> None:
        """
        Gaussian Error Linear Unit (GeLU) activation function.
        """

        super(GeLU, self).__init__()
        ActivationMethodAbstract.__init__(self)

    @override
    def _serialize(self) -> dict:
        return {}

    @classmethod
    def _deserialize(cls, **parameters: dict):
        return cls()


class ActivationMethodConstructors(Enum):
    """
    Enumeration of activation method constructors.
    """

    GELU = GeLU

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
