from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

import torch


class DataPointInputAbstract(ABC):
    """
    Data input for the neural network.
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


class DataPointOutputAbstract(ABC):
    """
    Data output for the neural network.
    """

    @property
    @abstractmethod
    def vector(self) -> torch.Tensor:
        """
        Convert the output to a torch.Tensor vector

        Returns
        -------
        torch.Tensor
            The output as a torch.Tensor.
        """


@dataclass(frozen=True)
class DataPoint:
    """
    Dataset for the neural network.

    Attributes
    ----------
    input: DatasetInput
        The input data.
    output: DatasetOutput
        The output data.
    """

    input: DataPointInputAbstract
    output: DataPointOutputAbstract


@dataclass(frozen=True)
class DataSet(torch.utils.data.Dataset):
    """
    Dataset for the neural network.

    Attributes
    ----------
    data_points: list[DataPoint]
        List of data points.
    input_labels: list[str]
        Labels for the x-axis. The length of which should match the length of [DataPointInput.vector].
    output_labels: list[str]
        Labels for the y-axis. The length of which should match the length of [DataPointOutput.vector].
    """

    input_labels: list[str]
    output_labels: list[str]
    _data_points: list[DataPoint] = field(init=False, default=None)

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
        return torch.stack([data_point.input.vector for data_point in self._data_points])

    @property
    def outputs(self) -> torch.Tensor:
        """
        Get all the outputs in the dataset.

        Returns
        -------
        list[torch.Tensor]
            List of all the outputs.
        """
        # TODO : Test this function
        return torch.stack([data_point.output.vector for data_point in self._data_points])

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
