from dataclasses import dataclass, field
from typing import Self

import torch

from .activation_methods import ActivationMethodAbstract
from .loss_methods import LossFunctionAbstract, LossFunctionConstructors
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
    learning_rate : float
        Learning rate for the optimizer.
    dropout_probability : float
        Dropout probability used in the network.
    use_batch_norm : bool
        Flag indicating if batch normalization is used.
    """

    model_name: str
    batch_size: int
    hidden_layers_node_count: tuple[int, ...]
    activations: tuple[ActivationMethodAbstract, ...]
    learning_rate: float
    dropout_probability: float
    use_batch_norm: bool

    # These are internal attributes which are properly set in [initialize] method
    is_initialized: bool = field(init=False, default=False)
    prediction_model: "_NeuralNetworkPredictionModel" = field(init=False, default=None)
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
        # TODO : Test this function

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
        activations: tuple[ActivationMethodAbstract, ...],
        batch_size: int = 64,
        learning_rate: float = 1e-2,
        dropout_probability: float = 0.2,
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
            learning_rate=learning_rate,
            dropout_probability=dropout_probability,
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
        # TODO : Test this function
        return self._prediction_model

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
        # TODO : Test this function
        super(_NeuralNetworkPredictionModel, self).__init__()
        layers_node_count = (
            (input_layer_node_count,) + neural_network_model.hidden_layers_node_count + (output_layer_node_count,)
        )
        activations = neural_network_model.activations
        dropout_probability = neural_network_model.dropout_probability
        use_batch_norm = neural_network_model.use_batch_norm

        # Initialize the layers of the neural network
        layers = torch.nn.ModuleList()
        for i in range(len(layers_node_count) - 1):
            layers.append(torch.nn.Linear(layers_node_count[i], layers_node_count[i + 1]))
            if i < len(layers_node_count) - 2:
                # Do not add activation and dropout for the output layer
                if use_batch_norm:
                    layers.append(torch.nn.BatchNorm1d(layers_node_count[i + 1]))
                layers.append(activations[i])
                layers.append(torch.nn.Dropout(dropout_probability))
        self._forward_model = torch.nn.Sequential(*layers)
        # self._forward_model.double()  # TODO RENDU ICI

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
        # TODO : Test this function
        return self._forward_model(x)
