import logging
from time import time
from typing import override

import biorbd
import numpy as np
import torch

from .biorbd_data_set import DataPointInputBiorbd, DataPointOutputBiorbd
from .biorbd_output_modes import BiorbdOutputModes
from ..reference_model_abstract import ReferenceModelAbstract
from ...prediction_model.data_set import DataSet, DataPoint

_logger = logging.getLogger(__name__)


class ReferenceModelBiorbd(ReferenceModelAbstract):

    def __init__(
        self,
        biorbd_model_path: str,
        muscle_names: tuple[str, ...],
        with_noise: bool = True,
        output_mode: BiorbdOutputModes = BiorbdOutputModes.TORQUE_MUS_DLMT_DQ,
        muscle_tendon_length_normalization: float = 1.0 * 100.0,
        muscle_tendon_lengths_jacobian_normalization: float = 10.0 * 100.0,
        muscle_forces_normalization: float = 0.001 * 100.0,
        tau_normalization: float = 0.01 * 100.0,
    ) -> None:
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
        output_mode: BiorbdOutputModes
            The output mode of the model
        """
        super(ReferenceModelBiorbd, self).__init__(with_noise=with_noise)

        self._model = biorbd.Model(biorbd_model_path)
        self._muscle_names = muscle_names
        self._output_mode = output_mode

        self._muscle_tendon_length_scale = muscle_tendon_length_normalization
        self._muscle_tendon_lengths_jacobian_scale = muscle_tendon_lengths_jacobian_normalization
        self._muscle_forces_scale = muscle_forces_normalization
        self._tau_scale = tau_normalization

    @property
    @override
    def name(self) -> str:
        return f"Biorbd_{self._output_mode.name}"

    @property
    @override
    def input_labels(self) -> tuple[str]:
        # TODO : Test this function
        return (*self.muscle_names, *self.q_names, *self.qdot_names)

    @property
    @override
    def output_labels(self) -> tuple[str]:
        # TODO : Test this function
        muscle_tendon_lengths_names = [f"{muscle_name}_tendon_length" for muscle_name in self.muscle_names]
        muscle_tendon_length_jacobian_names = []
        for muscle_name in self.muscle_names:
            muscle_tendon_length_jacobian_names += [f"{muscle_name}_{q_name}" for q_name in self.q_names]
        muscle_force_names = [f"{muscle_name}_force" for muscle_name in self.muscle_names]
        tau_names = self.tau_names

        return (*muscle_tendon_lengths_names, *muscle_tendon_length_jacobian_names, *muscle_force_names, *tau_names)

    @override
    def input_vector_to_coordinates(self, input: torch.Tensor) -> DataPointInputBiorbd:
        # TODO : Test this function
        return DataPointInputBiorbd(
            activations=input[: self.muscle_count],
            q=input[self.muscle_count : self.muscle_count + self._model.nbQ()],
            qdot=input[self.muscle_count + self._model.nbQ() :],
        )

    @override
    def output_vector_to_coordinates(self, output: torch.Tensor) -> DataPointOutputBiorbd:
        # TODO : Test this function
        n_mus = self.muscle_count
        n_q = self.q_count
        n_tau = self.tau_count
        if output.shape[0] != n_mus + n_mus * n_q + n_mus + n_tau:
            raise ValueError(
                f"The output vector should have a length of {n_mus + n_mus * n_q + n_mus + n_tau}, "
                f"but has a length of {output.shape[0]}."
            )

        muscle_tendon_lengths = output[:n_mus]
        muscle_tendon_lengths_jacobian = output[n_mus : n_mus + n_mus * n_q].reshape(n_mus, n_q)
        muscle_forces = output[n_mus + n_mus * n_q : n_mus + n_mus * n_q + n_mus]
        tau = output[n_mus + n_mus * n_q + n_mus :]

        return DataPointOutputBiorbd(
            muscle_tendon_lengths=muscle_tendon_lengths,
            muscle_tendon_lengths_jacobian=muscle_tendon_lengths_jacobian,
            muscle_forces=muscle_forces,
            tau=tau,
        )

    @override
    def generate_dataset(self, data_point_count: int) -> DataSet:
        # TODO : Test this function
        # Extract the min and max for each q, qdot and activations
        q_ranges = np.array([(q_range[0], q_range[1]) for q_range in self._get_q_ranges()]).T
        qdot_ranges = np.array([(-10.0 * np.pi, 10.0 * np.pi) for _ in range(self._model.nbQdot())]).T
        activations_ranges = np.array([(0.0, 1.0) for _ in range(self.muscle_count)]).T

        # Generate a data set by selecting randomly combinations of q, qdot, and activations
        _logger.info(f"Generating a dataset of {data_point_count} data points. This may take a while...")
        tic = time()
        data_set = DataSet(input_labels=self.input_labels, output_labels=self.output_labels)
        for _ in range(data_point_count):
            # Generate random data from the q, qdot, and activations ranges
            data_point_input = DataPointInputBiorbd(
                activations=torch.tensor(np.random.uniform(*activations_ranges)),
                q=torch.tensor(np.random.uniform(*q_ranges)),
                qdot=torch.tensor(np.random.uniform(*qdot_ranges)),
            )

            # Compute a data point from this input
            data_point_output = self._compute_data_point_output(data_point_input)
            data_set.append(DataPoint(input=data_point_input, target=data_point_output))

        _logger.info(f"Dataset generated in {time() - tic:.2f} seconds.")
        return data_set

    @property
    def q_count(self) -> int:
        """
        Get the number of degrees of freedom in the model.

        Returns
        -------
        int
            Number of degrees of freedom in the model.
        """
        # TODO : Test this function
        return self._model.nbQ()

    @property
    def q_names(self) -> tuple[str, ...]:
        """
        Get the names of the degrees of freedom in the model.

        Returns
        -------
        tuple[str, ...]
            Tuple of degree of freedom names.
        """
        # TODO : Test this function
        return tuple(f"q_{name.to_string().lower()}" for name in self._model.nameDof())

    @property
    def qdot_count(self) -> int:
        """
        Get the number of generalized velocities in the model.

        Returns
        -------
        int
            Number of generalized velocities in the model.
        """
        # TODO : Test this function
        return self._model.nbQdot()

    @property
    def qdot_names(self) -> tuple[str, ...]:
        """
        Get the names of the generalized velocities in the model.

        Returns
        -------
        tuple[str, ...]
            Tuple of generalized velocity names.
        """
        # TODO : Test this function
        return tuple(f"qdot_{name.to_string().lower()}" for name in self._model.nameDof())

    @property
    def tau_count(self) -> int:
        """
        Get the number of generalized forces in the model.

        Returns
        -------
        int
            Number of generalized forces in the model.
        """
        # TODO : Test this function
        return self._model.nbGeneralizedTorque()

    @property
    def tau_names(self) -> tuple[str, ...]:
        """
        Get the names of the generalized forces in the model.

        Returns
        -------
        tuple[str, ...]
            Tuple of generalized force names.
        """
        # TODO : Test this function
        return tuple(f"tau_{name.to_string().lower()}" for name in self._model.nameDof())

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
    def muscle_count(self) -> int:
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

    def _compute_data_point_output(self, data_point_input: DataPointInputBiorbd) -> DataPointOutputBiorbd:
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
        activations = np.array(data_point_input.activations)
        q = np.array(data_point_input.q)
        qdot = np.array(data_point_input.qdot)

        # Update the internal state of the model
        updated_model = self._model.UpdateKinematicsCustom(q, qdot)
        self._model.updateMuscles(updated_model, q, qdot)
        muscle_states = self._model.stateSet()
        for i, biorbd_muscle_index in enumerate(self._muscle_indices):
            state: biorbd.State = muscle_states[biorbd_muscle_index]
            state.setActivation(activations[i])

        # Compute the outputs for each muscle
        muscle_tendon_lengths = np.ndarray(self.muscle_count)
        muscle_tendon_lengths_jacobian = np.ndarray((self.muscle_count, self._model.nbQ()))
        muscle_forces = np.ndarray(self.muscle_count)
        for index, biorbd_muscle_index in enumerate(self._muscle_indices):
            muscle: biorbd.Muscle = self._model.muscle(biorbd_muscle_index)

            muscle_tendon_lengths[index] = muscle.position().length()
            muscle_tendon_lengths_jacobian[index, :] = muscle.position().jacobianLength().to_array()
            muscle_forces[index] = muscle.force(updated_model, q, qdot, muscle_states[biorbd_muscle_index])

        tau = -muscle_tendon_lengths_jacobian.T @ muscle_forces

        # Compute
        return DataPointOutputBiorbd(
            muscle_tendon_lengths=torch.tensor(muscle_tendon_lengths),
            muscle_tendon_lengths_jacobian=torch.tensor(muscle_tendon_lengths_jacobian),
            muscle_forces=torch.tensor(muscle_forces),
            tau=torch.tensor(tau),
        )

    @property
    @override
    def scaling_vector(self) -> torch.Tensor:
        # TODO : Test this function
        return torch.tensor(
            (
                [self._muscle_tendon_length_scale] * self.muscle_count
                + [self._muscle_tendon_lengths_jacobian_scale] * self.muscle_count * self.q_count
                + [self._muscle_forces_scale] * self.muscle_count
                + [self._tau_scale] * self.tau_count
            )
        )[None, :]
