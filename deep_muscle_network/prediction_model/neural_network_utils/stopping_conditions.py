from abc import ABC, abstractmethod
from enum import Enum
import logging
from typing import override

import torch

_logger = logging.getLogger(__name__)


class StoppingConditionsAbstract(ABC):
    @abstractmethod
    def should_stop(self, current_loss: float) -> bool:
        """
        Check if the current loss qualifies as an improvement. If too many epochs have passed without improvement,
        early stopping is triggered.

        Parameters
        ----------
        current_loss: float
            Current value of the monitored metric.

        Returns
        -------
        bool
            True if early stopping is triggered, otherwise False.
        """
        pass


class StoppingConditionMaxEpochs(StoppingConditionsAbstract):
    """
    Stop if the maximum number of epochs is reached.
    """

    def __init__(self, max_epochs: int):
        super(StoppingConditionMaxEpochs, self).__init__()

        self._max_epochs = max_epochs
        self._current_epoch = 0

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    @override
    def should_stop(self, current_loss: float) -> bool:
        self._current_epoch += 1
        if self._current_epoch >= self._max_epochs:
            _logger.info(f"Stopping training after maximum of {self._max_epochs} epochs reached.")
            return True
        return False


class StoppingConditionHasStoppedImproving(StoppingConditionsAbstract):
    """
    Early stopping mechanism to halt training when a monitored metric stops improving. This helps prevent overfitting
    by stopping training early if the model's performance on a validation set does not improve.

    Attributes
    ----------
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    epsilon : float
        Minimum change in the monitored metric to qualify as an improvement.
    """

    def __init__(self, patience: int = 50, epsilon: float = 1e-5):
        super(StoppingConditionHasStoppedImproving, self).__init__()

        self._patience = patience
        self.epsilon = epsilon

        self._lowest_loss_so_far = torch.inf
        self._epochs_without_improvement = 0
        self._current_epoch = 0

    @override
    def should_stop(self, current_loss: float) -> bool:
        self._current_epoch += 1

        if current_loss > self._lowest_loss_so_far - self.epsilon:
            self._epochs_without_improvement += 1
            if self._epochs_without_improvement >= self._patience:
                _logger.info(f"Early stopping triggered after {self._current_epoch} epochs.")
                return True
        else:
            self._lowest_loss_so_far = current_loss
            self._epochs_without_improvement = 0

        return False


class StoppingConditionConstructors(Enum):
    """
    Constructors for the stopping conditions used in the neural network model.
    """

    MAX_EPOCHS = StoppingConditionMaxEpochs
    HAS_STOPPED_IMPROVING = StoppingConditionHasStoppedImproving

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
