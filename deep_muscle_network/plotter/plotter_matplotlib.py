from math import sqrt
from functools import partial
from threading import Timer
from typing import Callable, override

from matplotlib import pyplot as plt
import numpy as np

from .plotter_abstract import PlotterAbstract
from ..prediction_model.data_set import DataSet


class PlotterMatplotlib(PlotterAbstract):
    def __init__(self, show_legends: bool = True):
        self._show_legends = show_legends

    @override
    def plot_prediction(self, data_set: DataSet, show_now: bool = True) -> plt.Figure | None:
        """
        Plot the predictions and targets of the data set.

        Parameters
        ----------
        data_set : DataSet
            The data set to plot the predictions and targets.
        show_now : bool, optional
            If True, the plot is shown now. Otherwise, it is returned.

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

        if show_now:
            plt.show()
            return None
        else:
            return fig

    @override
    def plot_loss_and_accuracy(self, data: list[tuple[float, float, float, float]]):
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
        x_data = np.arange(len(data))

        ax: plt.Axes = axs[0]
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss value")
        ax.set_title("Loss value")
        ax.plot(x_data, [value[0] for value in data], label="Training loss")
        ax.plot(x_data, [value[1] for value in data], label="Validation loss")
        ax.legend()

        ax: plt.Axes = axs[1]
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.plot(x_data, [value[2] for value in data], label="Training accuracy")
        ax.plot(x_data, [value[3] for value in data], label="Validation accuracy")
        ax.legend()

        # Update the plot
        plt.tight_layout()
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
