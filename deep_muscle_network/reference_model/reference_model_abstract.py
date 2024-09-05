from abc import ABC, abstractmethod

from ..prediction_model.data_set import DataSet


class ReferenceModelAbstract(ABC):
    def __init__(self, with_noise: bool) -> None:
        self._with_noise = with_noise

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
