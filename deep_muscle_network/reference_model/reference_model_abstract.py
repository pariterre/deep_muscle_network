from abc import ABC, abstractmethod

import torch

from ..prediction_model.data_set import DataSet, DataCoordinatesAbstract


class ReferenceModelAbstract(ABC):
    def __init__(self, with_noise: bool) -> None:
        self._with_noise = with_noise

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the reference model.
        """

    @property
    @abstractmethod
    def input_labels(self) -> tuple[str]:
        """
        Get the labels for the inputs. The labels should match the DataPointInput get from the [generate_data_set] method.

        Returns
        -------
        tuple[str]
            Labels for the inputs of the [generate_data_set] method.
        """

    @property
    @abstractmethod
    def output_labels(self) -> tuple[str]:
        """
        Get the labels for the outputs. The labels should match the DataPointOutput get from the [generate_data_set] method.

        Returns
        -------
        tuple[str]
            Labels for the outputs of the [generate_data_set] method.
        """

    @abstractmethod
    def input_vector_to_coordinates(self, input: torch.Tensor) -> DataCoordinatesAbstract:
        """
        Dispatch the input vector to the appropriate DataCoordinatesAbstract object.

        Parameters
        ----------
        input: torch.Tensor
            The input vector.

        Returns
        -------
        DataCoordinatesAbstract
            The DataCoordinatesAbstract object.
        """

    @abstractmethod
    def output_vector_to_coordinates(self, output: torch.Tensor) -> DataCoordinatesAbstract:
        """
        Dispatch the output vector to the appropriate DataCoordinatesAbstract object.

        Parameters
        ----------
        output: torch.Tensor
            The output vector.

        Returns
        -------
        DataCoordinatesAbstract
            The DataCoordinatesAbstract object.
        """

    @property
    def with_noise(self) -> bool:
        # TODO : Test this function
        return self._with_noise

    @abstractmethod
    def generate_dataset(self, data_point_count: int) -> DataSet:
        """
        Generate a reference dataset for the training of the prediction model.

        Parameters
        ----------
        data_point_count: int
            Number of DataPoint to generate. For each DataPoint, a set of DataPointInput is randomly generated, and the
            appropriate DataPointOutput is computed.
        """

    @property
    @abstractmethod
    def scaling_vector(self) -> torch.Tensor:
        """
        Get the scaling vector to normalize the output vector (normalization /=, denormalization *=).

        Returns
        -------
        torch.Tensor
            The scaling vector.
        """
