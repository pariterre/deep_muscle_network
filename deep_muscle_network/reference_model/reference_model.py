from abc import ABC, abstractmethod
from typing import Any

import torch

from ..prediction_model.neural_network_utils.data_set import DataSet, DataCoordinatesAbstract


class ReferenceModel(ABC):
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
    def generate_dataset(
        self, data_point_count: int, get_seed: bool = False, seed: Any = None
    ) -> DataSet | tuple[DataSet, Any]:
        """
        Generate a random dataset. If [get_seed] is True, the seed used to generate the data is returned along with
        the dataset as a second return value. The implementation of this method should include the capability to load
        data from the seed if it was already generated during the session, as it may very well occur that the same data
        is requested multiple times.

        Parameters
        ----------
        data_point_count: int
            Number of DataPoint to generate. For each DataPoint, a set of DataPointInput is randomly generated, and the
            appropriate DataPointOutput is computed.
        get_seed: bool
            If True, the seed used to generate the data is returned along with the dataset as a second return value. Said
            seed must be json serializable.
        seed: Any
            If not None, the data is generated using this seed, otherwise a random seed is used. If both [get_seed]
            and [seed] are provided, the [seed] is directly returned as the second return value as it will be the seed
            used to generate the data. The seed is the json serializable object used to generate the data.

        Returns
        -------
        DataSet
            The generated dataset.
        Any
            The seed used to generate the data. Only returned if [get_random_seed] is True.
        """

    @property
    @abstractmethod
    def scaling_vector(self) -> torch.Tensor:
        """
        Get the scaling vector to normalize the output vector (normalization *=, denormalization /=).
        The normalization should try to make the data to range between 1 and 100

        Returns
        -------
        torch.Tensor
            The scaling vector.
        """
