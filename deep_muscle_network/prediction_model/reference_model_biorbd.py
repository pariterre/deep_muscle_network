import logging
from time import time
from typing import override

import biorbd
import numpy as np

from .reference_model_abstract import ReferenceModelAbstract
from .data_point import DataSet, DataPoint, DataPointInput, DataPointOutput

_logger = logging.getLogger(__name__)


class ReferenceModelBiorbd(ReferenceModelAbstract):

    def __init__(self, biorbd_model_path: str, muscle_names: tuple[str, ...], with_noise: bool = True) -> None:
        """
        Constructor of the ReferenceModelBiorbd class.

        Parameters
        ----------
        biorbd_model_path: str
            Path to the biorbd model file.
        muscle_names: tuple[str, ...]
            Tuple of muscle names to used in the model, they should match existing muscles in the model.
        with_noise: bool
            If True, noise will be added to the data when generating the dataset using [generate_data_set].
        """
        super(ReferenceModelBiorbd, self).__init__(with_noise=with_noise)

        self._model = biorbd.Model(biorbd_model_path)
        self._muscle_names = muscle_names

    @override
    def generate_dataset(self, data_points_count: int) -> DataSet:
        # TODO : Test this function
        # Extract the min and max for each q, qdot and activations
        q_ranges = np.array([(q_range[0], q_range[1]) for q_range in self._get_q_ranges()]).T
        qdot_ranges = np.array([(-10.0 * np.pi, 10.0 * np.pi) for _ in range(self._model.nbQdot())]).T
        activations_ranges = np.array([(0.0, 1.0) for _ in range(self._muscle_count)]).T

        # Generate a data set by selecting randomly combinations of q, qdot, and activations
        _logger.info(f"Generating a dataset of {data_points_count} data points. This may take a while...")
        tic = time()
        data_set: DataSet = []
        for _ in range(data_points_count):
            # Generate random data from the q, qdot, and activations ranges
            data_point_input = DataPointInput(
                activations=np.random.uniform(*activations_ranges),
                q=np.random.uniform(*q_ranges),
                qdot=np.random.uniform(*qdot_ranges),
            )

            # Compute a data point from this input
            data_point_output = self._compute_data_point_output(data_point_input)
            data_set.append(DataPoint(input=data_point_input, output=data_point_output))

        _logger.info(f"Dataset generated in {time() - tic:.2f} seconds.")

    @property
    def muscle_names(self) -> tuple[str, ...]:
        """
        Get the names of the muscles used in the model.

        Returns
        -------
        tuple[str, ...]
            Tuple of muscle names.
        """
        # TODO : Test this function
        return self._muscle_names

    def _get_q_ranges(self) -> list[tuple[float, float]]:
        """
        Extracts the ranges of joint angles (q)

        Returns
        -------
        list[tuple[float, float]]
            Dictionary containing tuples (min, max) for each degree of freedom (q) in the model.
        """
        # TODO : Test this function
        q_ranges = []
        for segment_index in range(self._model.nbSegment()):
            # Get the range of motion for each degree of freedom in the segment
            q_ranges += [(ranges.min(), ranges.max()) for ranges in self._model.segment(segment_index).QRanges()]

        return q_ranges

    @property
    def _muscle_count(self) -> int:
        """
        Get the number of muscles in the model.

        Returns
        -------
        int
            Number of muscles in the model.
        """
        # TODO : Test this function
        return len(self._muscle_names)

    @property
    def _muscle_indices(self) -> tuple[int]:
        """
        Convert a tuple of muscle names to a tuple of muscle indices.

        Returns
        -------
        tuple[int]
            Tuple of muscle indices.
        """
        # TODO : Test this function
        biorbd_muscle_names = tuple(names.to_string() for names in self._model.muscleNames())
        return tuple(biorbd_muscle_names.index(muscle) for muscle in self._muscle_names)

    def _compute_data_point_output(self, data_point_input: DataPointInput) -> DataPointOutput:
        """
        Compute a new DataPoint.
        Note, it automatically updates the internal state of the model according to the q and qdot values.

        Parameters
        ----------
        inputs: DataPointInput
            The input data, consisting of muscle activations, q, and qdot.
        """
        # TODO : Test this function

        # Extract the inputs
        activations = data_point_input.activations
        q = data_point_input.q
        qdot = data_point_input.qdot

        # Update the internal state of the model
        updated_model = self._model.UpdateKinematicsCustom(q, qdot)
        self._model.updateMuscles(updated_model, q, qdot)
        muscle_states = self._model.stateSet()
        for i, biorbd_muscle_index in enumerate(self._muscle_indices):
            state: biorbd.State = muscle_states[biorbd_muscle_index]
            state.setActivation(activations[i])

        # Compute the outputs for each muscle
        muscle_tendon_lengths = np.ndarray(self._muscle_count)
        muscle_tendon_lengths_jacobian = np.ndarray((self._muscle_count, self._model.nbQ()))
        muscle_forces = np.ndarray(self._muscle_count)
        for index, biorbd_muscle_index in enumerate(self._muscle_indices):
            muscle: biorbd.Muscle = self._model.muscle(biorbd_muscle_index)

            muscle_tendon_lengths[index] = muscle.position().length()
            muscle_tendon_lengths_jacobian[index, :] = muscle.position().jacobianLength().to_array()
            muscle_forces[index] = muscle.force(updated_model, q, qdot, muscle_states[biorbd_muscle_index])

        tau = -muscle_tendon_lengths_jacobian.T @ muscle_forces

        # Compute
        return DataPointOutput(
            muscle_tendon_lengths=muscle_tendon_lengths,
            muscle_tendon_lengths_jacobian=muscle_tendon_lengths_jacobian,
            muscle_forces=muscle_forces,
            tau=tau,
        )
