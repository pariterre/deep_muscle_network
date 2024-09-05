from dataclasses import dataclass, field
from typing import Self

import torch

from .activation_methods import ActivationMethodsAbstract
from .loss_methods import LossMethodsAbstract, LossMethodConstructors
from ..utils.prediction_model_folder_structure import PredictionModelFolderStructure


@dataclass(frozen=True)
class NeuralNetworkModel:
    """
    Define the prediction model for the neural network model.

    Attributes
    ----------
    model_name : str
        Name of the model.
    batch_size : int
        Batch size used for training.
    hidder_layers_node_count : list[int]
        Number of nodes in the neural network, excluding the input and output layers
    activations : list[ActivationMethodsAbstract]
        List of activation functions used in the model, the length of which should be exactly the
        same as 'hidden_layers_node_count'.
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
    hidden_layers_node_count: tuple[int, ...]
    activations: tuple[ActivationMethodsAbstract, ...]
    l1_penalty: float
    l2_penalty: float
    learning_rate: float
    num_epochs: int
    criterion: LossMethodsAbstract
    dropout_prob: float
    use_batch_norm: bool

    # These are internal attributes which are properly set in [initialize] method
    is_initialized: bool = field(init=False, default=False)
    prediction_model: torch.nn.Module = field(init=False, default=None)
    optimizer: torch.optim.Optimizer = field(init=False, default=None)

    def __post_init__(self) -> None:
        # TODO : Test this function

        # Verify that activations, activation_names, and n_nodes have the same length
        if not (len(self.activations) == len(self.hidden_layers_node_count)):
            raise ValueError("The lengths of 'activations' and 'n_nodes' must be the same.")

    def initialize(self, input_layer_node_count: int, output_layer_node_count: int) -> None:
        """
        Initialize the prediction model.

        Parameters
        ----------
        input_layer_node_count : int
            Number of nodes in the input layer.
        output_layer_node_count : int
            Number of nodes in the output layer.
        """

        # Initialize the model
        object.__setattr__(
            self,
            "prediction_model",
            _NeuralNetworkPredictionModel(
                self, input_layer_node_count=input_layer_node_count, output_layer_node_count=output_layer_node_count
            ),
        )
        object.__setattr__(self, "optimizer", torch.optim.Adam(self.prediction_model.parameters(), self.learning_rate))
        object.__setattr__(self, "is_initialized", True)

    @classmethod
    def from_default(
        cls,
        model_name: str,
        hidden_layers_node_count: tuple[int, ...],
        activations: tuple[ActivationMethodsAbstract, ...],
        batch_size: int = 64,
        l1_penalty: float = 0.01,
        l2_penalty: float = 0.01,
        learning_rate: float = 1e-2,
        num_epochs: int = 1000,
        criterion: LossMethodsAbstract = LossMethodConstructors.MODIFIED_HUBER(delta=0.2, factor=1.0),
        dropout_prob: float = 0.2,
        use_batch_norm: bool = True,
    ) -> Self:
        """
        Return a generic prediction model from a default set of hyper parameters.

        Parameters
        ----------
        See the class attributes for the description of the parameters.

        Returns
        -------
        PredictionModel
            A default set of hyper parameters.
        """
        return cls(
            model_name=model_name,
            batch_size=batch_size,
            hidden_layers_node_count=hidden_layers_node_count,
            activations=activations,
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            criterion=criterion,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm,
        )

    @property
    def torch_model(self) -> torch.nn.Module:
        """
        Get the PyTorch model.

        Returns
        -------
        torch.nn.Module
            The PyTorch model.
        """
        return self._prediction_model

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
        file_path = folder.prediction_model_output_mode_path
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
        file_path = folder.prediction_model_output_mode_path
        raise NotImplementedError("The load method is not implemented yet.")


class _NeuralNetworkPredictionModel(torch.nn.Module):
    """
    Define the neural network model for the prediction model.
    """

    def __init__(
        self, neural_network_model: NeuralNetworkModel, input_layer_node_count: int, output_layer_node_count: int
    ):
        """
        Initialize the neural network model.

        Parameters
        ----------
        neural_network_model : NeuralNetworkModel
            The neural network model.
        input_layer_node_count : int
            Number of nodes in the input layer.
        output_layer_node_count : int
            Number of nodes in the output layer.
        """
        super(_NeuralNetworkPredictionModel, self).__init__()
        layers_node_count = (
            (input_layer_node_count,) + neural_network_model.hidden_layers_node_count + (output_layer_node_count,)
        )
        activations = neural_network_model.activations
        dropout_prob = neural_network_model.dropout_prob
        use_batch_norm = neural_network_model.use_batch_norm

        # Initialize the layers of the neural network
        layers = torch.nn.ModuleList()
        for i in range(len(layers_node_count) - 1):
            layers.append(torch.nn.Linear(layers_node_count[i], layers_node_count[i + 1]))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(layers_node_count[i + 1]))
            if i < len(layers_node_count) - 2:
                layers.append(activations[i])  # TODO : Should we have a value for the last activation?
            layers.append(torch.nn.Dropout(dropout_prob))
        self._forward_model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self._forward_model(x)
