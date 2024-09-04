from dataclasses import dataclass, field
from typing import Self

import torch

from .prediction_model_folder_structure import PredictionModelFolderStructure
from ..neural_network import (
    ActivationMethodsAbstract,
    LossMethodsAbstract,
    ActivationMethodConstructors,
    LossMethodConstructors,
)


@dataclass(frozen=True)
class HyperParametersModel:
    """
    Define the hyperparameters for the neural network model.

    Attributes
    ----------
    model_name : str
        Name of the model.
    batch_size : int
        Batch size used for training.
    n_nodes : list[int]
        Number of nodes in the neural network.
    activations : list[ActivationMethodsAbstract]
        List of activation functions used in the model.
    l1_penalty : float
        L1 regularization penalty value.
    l2_penalty : float
        L2 regularization penalty value.
    learning_rate : float
        Learning rate for the optimizer.
    num_epochs : int
        Number of epochs for training.
    criterion : LossMethodsAbstract
        Loss function used for training.
    dropout_prob : float
        Dropout probability used in the network.
    use_batch_norm : bool
        Flag indicating if batch normalization is used.
    """

    model_name: str
    batch_size: int
    n_nodes: tuple[int, ...]
    activations: tuple[ActivationMethodsAbstract, ...]
    l1_penalty: float
    l2_penalty: float
    learning_rate: float
    num_epochs: int
    criterion: LossMethodsAbstract
    dropout_prob: float
    use_batch_norm: bool
    optimizer: torch.optim.Optimizer = field(default=None)

    @classmethod
    def from_default(
        cls,
        model_name: str,
        batch_size: int = 64,
        n_nodes: tuple[int, ...] = (32, 32),
        activations: tuple[ActivationMethodsAbstract, ...] = (
            ActivationMethodConstructors.GELU(),
            ActivationMethodConstructors.GELU(),
        ),
        l1_penalty: float = 0.01,
        l2_penalty: float = 0.01,
        learning_rate: float = 1e-2,
        num_epochs: int = 1000,
        criterion: LossMethodsAbstract = LossMethodConstructors.MODIFIED_HUBER(delta=0.2, factor=1.0),
        dropout_prob: float = 0.2,
        use_batch_norm: bool = True,
    ) -> Self:
        """
        Return a generic default set of hyper parameters.

        Parameters
        ----------
        See the class attributes for the description of the parameters.

        Returns
        -------
        HyperParametersModel
            A default set of hyper parameters.
        """
        return cls(
            model_name=model_name,
            batch_size=batch_size,
            n_nodes=n_nodes,
            activations=activations,
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            criterion=criterion,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm,
        )

    def __post_init__(self) -> None:
        # Verify that activations, activation_names, and n_nodes have the same length
        if not (len(self.activations) == len(self.n_nodes)):
            raise ValueError("The lengths of 'activations' and 'n_nodes' must be the same.")

    def compute_optimiser(self, model) -> None:
        """
        Initialize the optimizer for the model using the Adam optimization algorithm.

        Parameters
        ----------
        model: torch.nn.Module
            The model for which the optimizer is to be computed.
        """
        # TODO : Move this to the __post_init__ method?
        # TODO : Deal with the frozen=True attribute
        # TODO : Test this method
        self.optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)

    def set_criterion(self, criterion: LossMethodsAbstract) -> None:
        """
        Set the loss criterion for the model.

        Parameters
        ----------
        criterion : LossMethodsAbstract
            The loss criterion to be set for the model.
        """
        # TODO : Deal with the frozen=True attribute (a copy_with method?)
        # TODO : Test this method
        self.criterion = criterion

    def save(self, folder: PredictionModelFolderStructure) -> None:
        """
        Save the model configuration to a file.

        Parameters
        ----------
        path : str
            Path to the file where the model configuration is to be saved.
        """

        # TODO : Implement the save method
        # TODO : Test this method
        file_path = folder.hyper_parameters_path
        raise NotImplementedError("The save method is not implemented yet.")

    @classmethod
    def load(cls, folder: PredictionModelFolderStructure) -> Self:
        """
        Load the model configuration from a file.

        Parameters
        ----------
        path : str
            Path to the file where the model configuration is saved.

        Returns
        -------
        HyperParametersModel
            The model configuration loaded from the file.
        """
        # TODO : Implement the load method
        # TODO : Test this method
        file_path = folder.hyper_parameters_path
        raise NotImplementedError("The load method is not implemented yet.")
