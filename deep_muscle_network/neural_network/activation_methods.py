from abc import ABC
from enum import Enum


import torch


class ActivationMethodsAbstract(torch.nn.Module, ABC):
    def __init__(self):
        super(ActivationMethodsAbstract, self).__init__()


class GeLU(ActivationMethodsAbstract):
    def __init__(self) -> None:
        """
        Gaussian Error Linear Unit (GeLU) activation function.
        """

        super(GeLU, self).__init__()


class ActivationMethodConstructors(Enum):
    """
    Enumeration of activation method constructors.
    """

    GELU = GeLU

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
