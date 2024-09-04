from abc import ABC, abstractmethod
import logging

from .data_point import DataSet


class ReferenceModelAbstract(ABC):
    def __init__(self, with_noise: bool) -> None:
        self._with_noise = with_noise

    @property
    def with_noise(self) -> bool:
        # TODO : Test this function
        return self._with_noise

    @abstractmethod
    def generate_dataset(self, data_points_count: int) -> DataSet:
        """
        Generate a reference dataset for the training of the prediction model.

        Parameters
        ----------
        data_points_count: int
            Number of DataPoint to generate. For each DataPoint, a set of DataPointInput is randomly generated, and the
            appropriate DataPointOutput is computed.
        """
