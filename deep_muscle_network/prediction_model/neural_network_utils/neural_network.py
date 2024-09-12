from dataclasses import dataclass, field
import json

import torch

from .activation_methods import ActivationMethodAbstract, ActivationMethodConstructors
from .loss_methods import LossFunctionAbstract, LossFunctionConstructors
from .neural_network_model import NeuralNetworkModel
from .neural_network_folder_structure import NeuralNetworkFolderStructure
from .stopping_conditions import StoppingConditionsAbstract, StoppingConditionConstructors


@dataclass
class NeuralNetwork:
    # Generic attributres
    training_data_count: int
    validation_data_count: int

    # Perceptron model
    hidden_layers_node_count: tuple[int, ...]
    use_batch_norm: bool = True
    activations: tuple[ActivationMethodAbstract, ...] = field(default=ActivationMethodConstructors.GELU())
    input_layer_node_count: int = field(init=False, default=None)
    output_layer_node_count: int = field(init=False, default=None)
    output_scaling_vector: torch.Tensor = field(init=False, default=None)

    # Learning attributes
    loss_function: LossFunctionAbstract = field(default=LossFunctionConstructors.MODIFIED_HUBER(delta=0.2, factor=1.0))
    stopping_conditions: tuple[StoppingConditionsAbstract, ...] = field(
        default=(
            StoppingConditionConstructors.MAX_EPOCHS(max_epochs=1000),
            StoppingConditionConstructors.HAS_STOPPED_IMPROVING(patience=50, epsilon=1e-5),
        )
    )
    learning_rate: float = 1e-2
    dropout_probability: float = 0.2

    # These are internal attributes which are properly set in [initialize] method
    is_initialized: bool = field(init=False, default=False)
    model: NeuralNetworkModel = field(init=False, default=None)
    optimizer: torch.optim.Optimizer = field(init=False, default=None)

    def __post_init__(self) -> None:
        # TODO : Test this function

        # Dispatch activations if needed
        if isinstance(self.activations, ActivationMethodAbstract):
            object.__setattr__(self, "activations", (self.activations,) * len(self.hidden_layers_node_count))

        # Verify that activations and hidden_layers_node_count have the same length
        if not (len(self.activations) == len(self.hidden_layers_node_count)):
            raise ValueError("The lengths of 'activations' and 'n_nodes' must be the same.")

    def set_reference_values(
        self, input_layer_node_count: int, output_layer_node_count: int, output_scaling_vector: torch.Tensor
    ) -> None:
        object.__setattr__(self, "input_layer_node_count", input_layer_node_count)
        object.__setattr__(self, "output_layer_node_count", output_layer_node_count)
        object.__setattr__(self, "output_scaling_vector", output_scaling_vector)
        object.__setattr__(self, "model", NeuralNetworkModel(self))
        object.__setattr__(self, "optimizer", torch.optim.Adam(self.model.parameters(), self.learning_rate))
        object.__setattr__(self, "is_initialized", True)

    def save(self, base_folder: NeuralNetworkFolderStructure, model_name: str) -> None:
        """
        Save the hyper parameters to a file.

        Parameters
        ----------
        base_folder : PredictionModelFolderStructure
            The folder structure where the hyper parameters will be saved.
        model_name : str
            Name of the model.
        """
        model_file_path = base_folder.trained_model_path(model_name=model_name)
        torch.save(self.model.state_dict(), model_file_path)

        parameters = {
            "training_data_count": self.training_data_count,
            "validation_data_count": self.validation_data_count,
            "use_batch_norm": self.use_batch_norm,
            "activations": tuple(activation.serialize() for activation in self.activations),
            "input_layer_node_count": self.input_layer_node_count,
            "hidden_layers_node_count": self.hidden_layers_node_count,
            "output_layer_node_count": self.output_layer_node_count,
            "output_scaling_vector": self.output_scaling_vector.tolist(),
            "loss_function": self.loss_function.serialize(),
            "stopping_conditions": tuple(
                stopping_condition.serialize() for stopping_condition in self.stopping_conditions
            ),
            "learning_rate": self.learning_rate,
            "dropout_probability": self.dropout_probability,
        }
        hyper_parameters_file_path = base_folder.hyper_parameters_model_path(model_name=model_name)
        with open(hyper_parameters_file_path, "w") as file:
            json.dump(parameters, file, indent=2)

    @classmethod
    def load(cls, base_folder: NeuralNetworkFolderStructure, model_name: str):
        """
        Load the hyper parameters from a file.

        Parameters
        ----------
        base_folder : PredictionModelFolderStructure
            The folder structure where the hyper parameters will be loaded.
        model_name : str
            Name of the model.
        """

        with open(base_folder.hyper_parameters_model_path(model_name=model_name)) as file:
            parameters: dict = json.load(file)

        neural_network = cls(
            training_data_count=parameters["training_data_count"],
            validation_data_count=parameters["validation_data_count"],
            use_batch_norm=parameters["use_batch_norm"],
            activations=tuple(
                ActivationMethodAbstract.deserialize(serialized_activation)
                for serialized_activation in parameters["activations"]
            ),
            hidden_layers_node_count=tuple(parameters["hidden_layers_node_count"]),
            loss_function=LossFunctionAbstract.deserialize(parameters["loss_function"]),
            stopping_conditions=tuple(
                StoppingConditionsAbstract.deserialize(serialized_stopping_condition)
                for serialized_stopping_condition in parameters["stopping_conditions"]
            ),
            learning_rate=parameters["learning_rate"],
            dropout_probability=parameters["dropout_probability"],
        )

        neural_network.set_reference_values(
            input_layer_node_count=parameters["input_layer_node_count"],
            output_layer_node_count=parameters["output_layer_node_count"],
            output_scaling_vector=torch.tensor(parameters["output_scaling_vector"]),
        )

        neural_network.model.load_state_dict(
            torch.load(base_folder.trained_model_path(model_name=model_name), weights_only=True)
        )

        return neural_network
