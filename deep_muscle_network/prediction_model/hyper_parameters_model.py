from dataclasses import dataclass, field
from typing import Self

import torch


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
    activations : list[torch.nn.Module]
        List of activation functions used in the model.
    activation_names : list[str]
        Names of the activation functions used.
    l1_penalty : float
        L1 regularization penalty value.
    l2_penalty : float
        L2 regularization penalty value.
    learning_rate : float
        Learning rate for the optimizer.
    num_epochs : int
        Number of epochs for training.
    criterion : torch.nn.Module
        Loss function used for training.
    dropout_prob : float
        Dropout probability used in the network.
    use_batch_norm : bool
        Flag indicating if batch normalization is used.
    """

    model_name: str
    batch_size: int
    n_nodes: tuple[int]
    activations: tuple[torch.nn.Module]
    activation_names: tuple[str]
    l1_penalty: float
    l2_penalty: float
    learning_rate: float
    num_epochs: int
    criterion: torch.nn.Module
    dropout_prob: float
    use_batch_norm: bool
    optimizer: torch.optim.optimizer.Optimizer = field(default=None)

    def __post_init__(self) -> None:
        # Verify that activations, activation_names, and n_nodes have the same length
        if not (len(self.activations) == len(self.activation_names) == len(self.n_nodes)):
            raise ValueError("The lengths of 'activations', 'activation_names', and 'n_nodes' must be the same.")

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

    def set_criterion(self, criterion_params) -> None:
        """
        Set the loss criterion for the model.

        Parameters
        ----------
        criterion_params (torch.nn.Module): The loss function to be used for training.
        """
        # TODO : Deal with the frozen=True attribute (a copy_with method?)
        # TODO : Test this method
        self.criterion = criterion_params

    def save(self, path) -> None:
        """
        Save the model configuration to a file.

        Parameters
        ----------
        path : str
            Path to the file where the model configuration is to be saved.
        """

        # TODO : Implement the save method
        # TODO : Test this method
        raise NotImplementedError("The save method is not implemented yet.")

    @classmethod
    def load(cls, path) -> Self:
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
        raise NotImplementedError("The load method is not implemented yet.")
