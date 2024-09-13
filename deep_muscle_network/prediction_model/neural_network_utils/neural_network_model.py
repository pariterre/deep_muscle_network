from typing import Protocol

import torch

from .activation_methods import ActivationMethodAbstract
from .torch_utils import get_torch_device


class NeuralNetwork(Protocol):
    """
    Neural network configuration for the prediction model.

    Attributes
    ----------
    use_batch_norm : bool
        If True, batch normalization is used.
    activations : tuple[ActivationMethodAbstract, ...]
        Activation functions to use for each layer.
    input_layer_node_count : int
        Number of nodes in the input layer.
    hidden_layers_node_count : tuple[int, ...]
        Number of nodes in each hidden layer.
    output_layer_node_count : int
        Number of nodes in the output layer.
    dropout_probability : float
        Dropout probability.
    optimizer : torch.optim.Optimizer
        Optimizer to use for training the neural network.
    """

    use_batch_norm: bool
    activations: tuple[ActivationMethodAbstract, ...]
    input_layer_node_count: int
    hidden_layers_node_count: tuple[int, ...]
    output_layer_node_count: int
    dropout_probability: float
    optimizer: torch.optim.Optimizer


class NeuralNetworkModel(torch.nn.Module):
    """
    Define the neural network model for the prediction model.
    """

    def __init__(self, neural_network: NeuralNetwork):
        """
        Initialize the neural network model.

        Parameters
        ----------
        neural_network : NeuralNetworkModel
            The neural network configuration to use to train the model.
        """
        # TODO : Test this function
        super(NeuralNetworkModel, self).__init__()
        first_and_hidden_layers_node_count = (
            neural_network.input_layer_node_count,
        ) + neural_network.hidden_layers_node_count
        activations = neural_network.activations
        dropout_probability = neural_network.dropout_probability
        use_batch_norm = neural_network.use_batch_norm

        # Initialize the layers of the neural network
        layers = torch.nn.ModuleList()
        for i in range(len(first_and_hidden_layers_node_count) - 1):
            layers.append(
                torch.nn.Linear(first_and_hidden_layers_node_count[i], first_and_hidden_layers_node_count[i + 1])
            )
            if use_batch_norm:
                torch.nn.BatchNorm1d(first_and_hidden_layers_node_count[i + 1])
            layers.append(activations[i])
            layers.append(torch.nn.Dropout(dropout_probability))
        layers.append(torch.nn.Linear(first_and_hidden_layers_node_count[-1], (neural_network.output_layer_node_count)))

        self._forward_model = torch.nn.Sequential(*layers)
        self._forward_model.double().to(get_torch_device())

        self.eval()

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
        output = torch.Tensor(x.shape[0], self._forward_model[-1].out_features)
        for i, data in enumerate(x):
            output[i, :] = self._forward_model(data)
        return output
