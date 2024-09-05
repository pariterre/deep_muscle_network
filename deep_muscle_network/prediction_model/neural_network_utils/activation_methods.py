from abc import ABC
from enum import Enum

import torch


class ActivationMethodAbstract(torch.nn.Module, ABC):
    def __init__(self):
        super(ActivationMethodAbstract, self).__init__()


class GeLU(ActivationMethodAbstract):
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
