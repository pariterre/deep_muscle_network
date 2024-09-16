from abc import ABC, abstractmethod

import torch

from ..prediction_model.neural_network_utils.data_set import DataSet
from ..prediction_model.neural_network_utils.training_data import TrainingData


class Plotter(ABC):
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

    @abstractmethod
    def pareto_front(
        self, x_data: torch.Tensor, y_data: torch.Tensor, title: str | None, x_label: str | None, y_label: str | None
    ) -> None:
        """
        Plot a pareto front from a list of values. Pareto front is the curve that represents the best trade-offs between
        two objectives.
        The [x_data, y_data] values are the points to plot and must match in size.

        Parameters
        ----------
        x_data : torch.Tensor
            The x values.
        y_data : torch.Tensor
            The y values.
        title : str, optional
            The title of the plot.
        x_label : str
            The label of the x-axis.
        y_label : str
            The label of the y-axis.
        """
