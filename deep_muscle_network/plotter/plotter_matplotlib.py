from math import sqrt
from typing import override

from matplotlib import pyplot as plt
import numpy as np
import torch

from .utils import get_pareto_front_indices
from .plotter import Plotter
from ..prediction_model.neural_network_utils.data_set import DataSet
from ..prediction_model.neural_network_utils.training_data import TrainingData


class PlotterMatplotlib(Plotter):
    def __init__(self, show_legends: bool = False):
        self._show_legends = show_legends

    def show(self):
        """
        Plot the figures in a blocking way.
        """
        plt.show()

    @override
    def plot_prediction(self, data_set: DataSet) -> plt.Figure | None:
        """
        Plot the predictions and targets of the data set.

        Parameters
        ----------
        data_set : DataSet
            The data set to plot the predictions and targets.

        Returns
        -------
        plt.Figure | None
            The figure containing the plot if show_now is False. Otherwise, None
        """

        data_count = data_set.output_len
        row_count, col_count = _compute_ideal_row_to_column_count_ratio(data_count)

        y_labels = data_set.output_labels
        targets = data_set.targets.cpu().numpy()
        predictions = data_set.predictions.detach().cpu().numpy()
        accuracy = data_set.predictions_accuracy
        relative_accuracy = data_set.relative_predictions_accuracy * 100
        precision = data_set.predictions_precision
        relative_precision = data_set.relative_predictions_precision * 100

        fig, axs = plt.subplots(row_count, col_count, figsize=(15, 10))
        axs: np.ndarray[plt.Axes]
        axs = axs.flatten()
        for i in range(data_count):
            ax: plt.Axes = axs[i] if data_count > 1 else axs

            ax.plot(targets[i, :], label="True values", marker="^", markersize=2)
            ax.plot(predictions[i, :], label="Predictions", marker="o", linestyle="--", markersize=2)

            ax.set_xlabel("Sample")
            ax.set_ylabel("Value")
            ax.set_title(
                f"{y_labels[i]}\n"
                f"acc = {accuracy[i]:.6f} ({relative_accuracy[i]:.3f}%)\n"
                f"precision = {precision[i]:.6f} ({relative_precision[i]:.3f}%)",
                fontsize="smaller",
            )
            if self._show_legends:
                ax.legend()

        fig.suptitle(f"Predictions and targets", fontweight="bold")
        plt.tight_layout()
        plt.show(block=False)

    @override
    def plot_loss_and_accuracy(self, data: TrainingData):
        """
        Plot the training and validation loss and accuracy during training.

        Parameters
        ----------
        data : list[tuple[float, float, float, float]]
            The data to plot. Each tuple contains the training loss, validation loss, training accuracy, and validation accuracy.
        """
        # TODO : Test this function

        # Prepare the figure
        fig, axs = plt.subplots(2, 1)
        fig.suptitle("Training and validation loss during training", fontweight="bold")

        # Plot the data
        x_data = np.arange(data.epoch_count)

        ax: plt.Axes = axs[0]
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss value")
        ax.set_title("Loss value")
        ax.plot(x_data, data.training_loss, label="Training loss")
        ax.plot(x_data, data.validation_loss, label="Validation loss")
        ax.legend()

        ax: plt.Axes = axs[1]
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.plot(x_data, data.training_accuracy, label="Training accuracy")
        ax.plot(x_data, data.validation_accuracy, label="Validation accuracy")
        ax.legend()

        # Update the plot
        plt.tight_layout()
        plt.show(block=False)

    def pareto_front(
        self, x_data: torch.Tensor, y_data: torch.Tensor, title: str | None, x_label: str | None, y_label: str | None
    ) -> None:
        if len(x_data) != len(y_data):
            raise ValueError("The x and y data must have the same length.")

        # Generate unique colors for each point using a colormap
        point_count = len(x_data)
        colors = plt.cm.jet(np.linspace(0, 1, point_count))  # Use the jet colormap for diversity

        # Configure plot size and scale
        plt.figure(figsize=(10, 5))
        plt.xscale("log")
        plt.yscale("log")

        # Detect Pareto front points
        pareto_front_indices = get_pareto_front_indices(x_data=x_data, y_data=y_data)

        # Plot each point and highlight Pareto front points
        for i in range(point_count):
            plt.scatter(x_data[i], y_data[i], marker="P", color=colors[i])  # Plot each point

            # Highlight and annotate Pareto front points
            if i in pareto_front_indices:
                plt.scatter(x_data[i], y_data[i], edgecolor="black", facecolor="none", s=100)
                plt.text(x_data[i], y_data[i], i, fontsize=9, ha="right", weight="bold")

        # Draw a line connecting Pareto front points
        pareto_points = sorted([(x_data[i], y_data[i]) for i in pareto_front_indices])
        pareto_x, pareto_y = zip(*pareto_points)  # Unzip into x and y components
        plt.plot(pareto_x, pareto_y, linestyle="--", color="black", alpha=0.6, label="Pareto_front")

        # Label axes and set plot title
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        if title is not None:
            plt.title(title)

        plt.grid(True)
        plt.show(block=False)


def _compute_ideal_row_to_column_count_ratio(element_count: int) -> tuple[int, int]:
    """
    Compute ideal row and col for subplots

    Parameters
    ----------
    sum : int
        sum of all number of plot to do in the figure

    Returns
    -------
    row : int
        number of row for subplot
    col : int
        number of col for subplot
    """

    #  div : int, number of col max
    div = round(sqrt(element_count))

    col_count = element_count // div
    if element_count % div != 0:
        col_count += 1

    row_count = min(element_count, div)

    return row_count, col_count
