from dataclasses import dataclass

import numpy as np


# TODO : Inherit from a common interface from [neural_networks] package


@dataclass(frozen=True)
class DataPointInput:
    """
    Data input for the neural network.

    Attributes
    ----------
    activations: np.ndarray
        Muscle activations.
    q: np.ndarray
        The vector of generalized coordinates.
    qdot: np.ndarray
        The vector of generalized velocities.
    """

    activations: np.ndarray
    q: np.ndarray
    qdot: np.ndarray


@dataclass(frozen=True)
class DataPointOutput:
    """
    Data output for the neural network.

    Attributes
    ----------
    muscle_tendon_lengths: np.ndarray
        The vector of muscle tendon lengths for each muscle.
    muscle_tendon_lengths_jacobian: np.ndarray
        The jacobian matrix of muscle tendon lengths relative to the generalized coordinates.
    muscle_forces: np.ndarray
        The vector of muscle forces for each muscle.
    tau: np.ndarray
        The vector of generalized forces.
    """

    muscle_tendon_lengths: np.ndarray
    muscle_tendon_lengths_jacobian: np.ndarray
    muscle_forces: np.ndarray
    tau: np.ndarray


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

    input: DataPointInput
    output: DataPointOutput


@dataclass(frozen=True)
class DataSet:
    """
    Dataset for the neural network.

    Attributes
    ----------
    data_points: list[DataPoint]
        List of data points.
    """

    data_points: list[DataPoint]

    def append(self, data_point: DataPoint) -> None:
        """
        Append a data point to the dataset.

        Parameters
        ----------
        data_point: DataPoint
            Data point to append.
        """
        self.data_points.append(data_point)

    def set_training_set_size():
        # TODO RENDU ICI!! (make set_training_size or set_validation_size)
        pass

    def training_set():
        # TODO : Once the set size is set, this method cas select and cache the training set AND the validation set
        pass

    def validation_set():
        # TODO : Once the set size is set, this method cas select and cache the training set AND the validation set
        pass
