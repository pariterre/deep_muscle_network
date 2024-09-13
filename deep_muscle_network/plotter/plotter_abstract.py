from abc import ABC, abstractmethod
from typing import Callable

from ..prediction_model.neural_network_utils.data_set import DataSet
from ..prediction_model.neural_network_utils.training_data import TrainingData


class PlotterAbstract(ABC):
    @abstractmethod
    def plot_prediction(self, data_set: DataSet):
        """
        Plot the data.

        Parameters
        ----------
        data_set: DataSet
            The data set to plot.
        """

    @abstractmethod
    def plot_loss_and_accuracy(self) -> TrainingData:
        """
        Prepare the online training. Returns a callback function that should be called at each epoch that will update the
        plot.
        This callback waits a list of tuples[float, float, float, float]. Each tuple contains the
            - training loss value (float)
            - validation loss value (float)
            - training accuracy (float)
            - validation accuracy (float)
        """
