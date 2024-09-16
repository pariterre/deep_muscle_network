from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

import torch


class DataCoordinatesAbstract(ABC):
    """
    Data coordinates that composes a data point (i.e. x-y coordinates) for the neural network.
    """

    @property
    @abstractmethod
    def vector(self) -> torch.Tensor:
        """
        Get the input as a torch.Tensor vector.

        Returns
        -------
        torch.Tensor
            The input as a torch.Tensor.
        """


@dataclass(frozen=True)
class DataPoint:
    """
    Dataset for the neural network.

    Attributes
    ----------
    input: DataCoordinatesAbstract
        The input data.
    target: DataCoordinatesAbstract | None
        The target data if available for supervised learning.
    predictions: DataCoordinatesAbstract | None
        The predictions data if available computed by the model.
    """

    input: DataCoordinatesAbstract
    target: DataCoordinatesAbstract | None = None
    prediction: DataCoordinatesAbstract | None = None


@dataclass(frozen=True)
class DataSet(torch.utils.data.Dataset):
    """
    Dataset for the neural network.

    Attributes
    ----------
    data_points: list[DataPoint]
        List of data points.
    input_labels: list[str]
        Labels for the x-axis. The length of which should match the length of input.
    output_labels: list[str]
        Labels for the y-axis. The length of which should match the length of target and prediction.
    """

    input_labels: list[str]
    output_labels: list[str]
    _data_points: list[DataPoint] = field(init=False, default=None)
    _is_predictions_filled: bool = field(init=False, default=False)

    def __post_init__(self):
        # TODO : Test this function
        object.__setattr__(self, "_data_points", [])

    def append(self, data_point: DataPoint) -> None:
        """
        Append a data point to the dataset.

        Parameters
        ----------
        data_point: DataPoint
            Data point to append.
        """
        # TODO : Test this function
        self._data_points.append(data_point)

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        """
        # TODO : Test this function
        return len(self._data_points)

    @property
    def inputs(self) -> torch.Tensor:
        """
        Get all the inputs in the dataset.

        Returns
        -------
        list[torch.Tensor]
            List of all the inputs.
        """
        # TODO : Test this function
        return torch.stack([data_point.input.vector for data_point in self._data_points]).T

    @property
    def targets(self) -> torch.Tensor:
        """
        Get all the target in the dataset.

        Returns
        -------
        list[torch.Tensor]
            List of all the targets.
        """
        # TODO : Test this function
        return torch.stack([data_point.target.vector for data_point in self._data_points]).T

    def fill_predictions(self, prediction_model, reference_model) -> None:
        """
        Fill the prediction field of the DataPoints using the predictions from the predicting_model and dispatched according
        to the reference model

        Parameters
        ----------
        prediction_model: PredictionModel
            The prediction model to use to predict the data.
        reference_model: ReferenceModelAbstract
            The reference model to use to dispatch the predictions
        """
        # TODO : Test this function
        from .. import PredictionModel
        from ...reference_model import ReferenceModel

        if not isinstance(prediction_model, PredictionModel):
            raise TypeError("The prediction_model should be an instance of PredictionModel.")

        if not isinstance(reference_model, ReferenceModel):
            raise TypeError("The reference_model should be an instance of ReferenceModelAbstract.")

        predictions = prediction_model.predict(data_set=self)
        for data_point, prediction in zip(self._data_points, predictions.T):
            coordinates = reference_model.output_vector_to_coordinates(output=prediction)
            object.__setattr__(data_point, "prediction", coordinates)

        object.__setattr__(self, "_is_predictions_filled", True)

    @property
    def predictions(self) -> torch.Tensor:
        """
        Get all the predictions in the dataset.

        Returns
        -------
        list[torch.Tensor]
            List of all the predictions.
        """
        # TODO : Test this function
        if not self._is_predictions_filled:
            raise ValueError(
                "The predictions have not been filled yet. Please call the [fill_predictions] method first."
            )

        return torch.stack([data_point.prediction.vector for data_point in self._data_points]).T

    @property
    def predictions_error(self) -> torch.Tensor:
        """
        Get the error of the predictions (i.e. the difference between the predictions and the targets).
        Positive values indicate that the prediction is higher than the target.
        This method assumes that the predictions and the targets are filled.

        Returns
        -------
        torch.Tensor
            The error of the predictions.
        """
        # TODO : Test this function
        return self.predictions - self.targets

    @property
    def absolute_predictions_error(self) -> torch.Tensor:
        """
        Get the absolute error of the predictions (i.e. the difference between the predictions and the targets).
        This method assumes that the predictions and the targets are filled.

        Note : this corresponds to the accuracy of the predictions.

        Parameters
        ----------
        compute_mean : bool
            If True, the mean of the absolute error will be computed.

        Returns
        -------
        torch.Tensor
            The error of the predictions.
        """
        # TODO : Test this function
        return self.predictions_error.abs()

    @property
    def relative_predictions_error(self) -> torch.Tensor:
        """
        Get the relative error of the predictions (i.e. the relative difference between the predictions and the targets
        over the targets).
        This method assumes that the predictions and the targets are filled.

        Returns
        -------
        torch.Tensor
            The relative error of the predictions.
        """
        # TODO : Test this function
        return self.predictions_error / self.targets

    @property
    def absolute_relative_predictions_error(self) -> torch.Tensor:
        """
        Get the absolute relative error of the predictions (i.e. the absolute relative difference between the predictions
        and the targets over the targets).
        This method assumes that the predictions and the targets are filled.

        Returns
        -------
        torch.Tensor
            The absolute relative error of the predictions.
        """
        # TODO : Test this function
        return self.relative_predictions_error.abs()

    @property
    def predictions_accuracy(self) -> torch.Tensor:
        """
        The mean of the absolute error of the predictions.
        This method assumes that the predictions and the targets are filled.

        Returns
        -------
        torch.Tensor
            The accuracy of the predictions for each output.
        """
        # TODO : Test this function
        return self.absolute_predictions_error.mean(axis=1)

    @property
    def relative_predictions_accuracy(self) -> torch.Tensor:
        """
        The mean of the relative error of the predictions.
        This method assumes that the predictions and the targets are filled.

        Returns
        -------
        torch.Tensor
            The relative accuracy of the predictions for each output.
        """
        # TODO : Test this function
        return self.absolute_relative_predictions_error.mean(axis=1)

    @property
    def predictions_precision(self) -> torch.Tensor:
        """
        Get the precision of the predictions (i.e. the root mean square error of the predictions accuracy).
        This method assumes that the predictions and the targets are filled.

        Returns
        -------
        torch.Tensor
            The precision of the predictions for each output.
        """
        # TODO : Test this function
        return (self.predictions_error**2).mean(axis=1).sqrt()

    @property
    def relative_predictions_precision(self) -> torch.Tensor:
        """
        Get the precision of the predictions (i.e. the standard deviation of the predictions over the mean measured values).
        This method assumes that the predictions and the targets are filled.

        Returns
        -------
        torch.Tensor
            The precision of the predictions for each output.
        """
        # TODO : Test this function
        return self.predictions.std(axis=1) / self.targets.mean(axis=1)

    @property
    def input_len(self) -> int:
        """
        Get the length of the input.

        Returns
        -------
        int
            Length of the input.
        """
        # TODO : Test this function
        return len(self.input_labels)

    @property
    def output_len(self) -> int:
        """
        Get the length of the output.

        Returns
        -------
        int
            Length of the output.
        """
        # TODO : Test this function
        return len(self.output_labels)

    def __getitem__(self, index: slice | int) -> Self:
        # TODO : Test this function
        if isinstance(index, int):
            index = slice(index, index + 1)

        data_set = DataSet(input_labels=self.input_labels, output_labels=self.output_labels)
        object.__setattr__(data_set, "_data_points", self._data_points[index])
        return data_set

    def __iter__(self):
        # TODO : Test this function
        for index in range(len(self._data_points)):
            yield self.__getitem__(index)

    def __repr__(self) -> str:
        # TODO : Test this function
        return (
            f"DataSet(\n"
            f"\tlen={len(self._data_points)}\n"
            f"\tinput_labels={self.input_labels}\n"
            f"\toutput_labels={self.output_labels}\n"
            f")"
        )
