from dataclasses import dataclass, field
from typing import override, Self

import torch

from ...prediction_model.data_set import DataCoordinatesAbstract

# TODO : Inherit from a common interface from [neural_networks] package


@dataclass(frozen=True)
class DataPointInputBiorbd(DataCoordinatesAbstract):
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

    activations: torch.Tensor
    q: torch.Tensor
    qdot: torch.Tensor
    _vector: torch.Tensor = field(init=False, default=None)

    def __post_init__(self):
        # TODO : Test this function

        # Check that the attributes don't include the time dimension (i.e., they are one-dimensional tensors)
        if not (len(self.activations.shape) == len(self.q.shape) == len(self.qdot.shape) == 1):
            raise ValueError("The activations, q, and qdot tensors should be one-dimensional tensors.")

        # Prepare the vector attribute
        object.__setattr__(self, "_vector", torch.cat((self.activations, self.q, self.qdot)))

    @property
    @override
    def vector(self) -> torch.Tensor:
        return self._vector


@dataclass(frozen=True)
class DataPointOutputBiorbd(DataCoordinatesAbstract):
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

    muscle_tendon_lengths: torch.Tensor
    muscle_tendon_lengths_jacobian: torch.Tensor
    muscle_forces: torch.Tensor
    tau: torch.Tensor
    _vector: torch.Tensor = field(init=False, default=None)

    def __post_init__(self):
        # TODO : Test this function

        # Check that the attributes don't include the time dimension (i.e., they are one-dimensional tensors)
        if not (len(self.muscle_tendon_lengths.shape) == len(self.muscle_forces.shape) == len(self.tau.shape) == 1):
            raise ValueError(
                "The muscle_tendon_lengths, muscle_forces, and tau tensors should be one-dimensional tensors."
            )

        # Check the same for the jacobian, but it should have two dimensions as it is a matrix
        if not len(self.muscle_tendon_lengths_jacobian.shape) == 2:
            raise ValueError("The muscle_tendon_lengths_jacobian matrix should have two dimensions.")
        if not self.muscle_tendon_lengths_jacobian.shape == (self.muscle_forces.shape[0], self.tau.shape[0]):
            raise ValueError(
                f"The muscle_tendon_lengths_jacobian matrix should have a shape of "
                f"(n_muscles={self.muscle_forces.shape[0]}, n_q={self.tau.shape[0]})."
            )

        # Prepare the vector attribute
        object.__setattr__(
            self,
            "_vector",
            torch.cat(
                (
                    self.muscle_tendon_lengths,
                    self.muscle_tendon_lengths_jacobian.flatten(),
                    self.muscle_forces,
                    self.tau,
                )
            ),
        )

    @property
    @override
    def vector(self) -> torch.Tensor:
        return self._vector
