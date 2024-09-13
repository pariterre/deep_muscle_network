from dataclasses import dataclass, field
import json
from typing import Any

import torch

from .activation_methods import ActivationMethodAbstract, ActivationMethodConstructors
from .loss_methods import LossFunctionAbstract, LossFunctionConstructors
from .neural_network_model import NeuralNetworkModel
from .neural_network_folder_structure import NeuralNetworkFolderStructure
from .torch_utils import get_torch_device
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
        object.__setattr__(self, "output_scaling_vector", output_scaling_vector.to(get_torch_device()))
        object.__setattr__(self, "model", NeuralNetworkModel(self))
        object.__setattr__(self, "optimizer", torch.optim.Adam(self.model.parameters(), self.learning_rate))
        object.__setattr__(self, "is_initialized", True)

    def serialize(self) -> dict[str, Any]:
        """
        Serialize the hyper parameters.

        Returns
        -------
        dict[str, Any]
            The serialized hyper parameters.
        """

        return {
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

    def has_a_trained_model(self, base_folder: NeuralNetworkFolderStructure) -> bool:
        """
        Check if a trained model exists.

        Parameters
        ----------
        base_folder : PredictionModelFolderStructure
            The folder structure where the trained model is saved.
        model_name : str
            Name of the model.

        Returns
        -------
        bool
            True if a trained model exists, False otherwise.
        """
        parameters = self.serialize()
        previous_trainings = _load_previous_training(base_folder=base_folder)
        key = _find_hyper_parameters_key_in_previous_trainings(parameters, previous_trainings)
        return key in previous_trainings

    def save(self, base_folder: NeuralNetworkFolderStructure, model_name: str) -> str:
        """
        Save the hyper parameters to a file.

        Parameters
        ----------
        base_folder : PredictionModelFolderStructure
            The folder structure where the hyper parameters will be saved.
        model_name : str
            Name of the model.

        Returns
        -------
        str
            The file name where the model is saved, whitout the extension.
        """

        parameters = self.serialize()
        previous_trainings = _load_previous_training(base_folder=base_folder)
        key = _find_hyper_parameters_key_in_previous_trainings(
            parameters=parameters, previous_trainings=previous_trainings
        )
        previous_trainings[key] = parameters

        hyper_parameters_file_path = base_folder.hyper_parameters_model_path
        with open(hyper_parameters_file_path, "w") as file:
            # Add the super key (name of the model)
            json.dump(previous_trainings, file, indent=2)

        model_file_name = f"{model_name}_{key}"
        model_file_path = base_folder.trained_model_path(model_name=model_file_name)
        torch.save(self.model.state_dict(), model_file_path)
        return model_file_name

    def load(self, base_folder: NeuralNetworkFolderStructure, model_name: str) -> None:
        """
        Load the hyper parameters from a file.

        Parameters
        ----------
        base_folder : PredictionModelFolderStructure
            The folder structure where the hyper parameters will be loaded.
        model_name : str
            Name of the model.
        """

        parameters = self.serialize()
        previous_trainings = _load_previous_training(base_folder=base_folder)
        key = _find_hyper_parameters_key_in_previous_trainings(parameters, previous_trainings)
        if key not in previous_trainings:
            raise FileNotFoundError(f"The model {model_name} has not been trained yet.")
        parameters = previous_trainings[key]

        self.model.load_state_dict(
            torch.load(base_folder.trained_model_path(model_name=f"{model_name}_{key}"), weights_only=True)
        )


def _load_previous_training(base_folder: NeuralNetworkFolderStructure) -> dict[str, Any]:
    """
    Load the previous training from a file.

    Parameters
    ----------
    base_folder : PredictionModelFolderStructure
        The folder structure where the previous training is saved.
    """

    try:
        with open(base_folder.hyper_parameters_model_path) as file:
            return json.load(file)
    except:
        return {}


def _find_hyper_parameters_key_in_previous_trainings(
    parameters: dict[str, Any], previous_trainings: dict[str, Any]
) -> str:
    """
    Find if the hyper parameters are identical to a previous training.

    Parameters
    ----------
    parameters : dict[str, Any]
        The hyper parameters to compare.
    previous_trainings : dict[str, Any]
        The previous trainings to compare with.

    Returns
    -------
    str
        The key of the identical hyper parameters. If no identical hyper parameters are found, a new keep is created.
    """

    for key, previous_parameters in previous_trainings.items():
        if parameters.keys() != previous_parameters.keys():
            continue

        has_differences = False
        for sub_key in parameters.keys():
            if isinstance(parameters[sub_key], (list, tuple)):
                if tuple(parameters[sub_key]) != tuple(previous_parameters[sub_key]):
                    has_differences = True
                    break
            else:
                if parameters[sub_key] != previous_parameters[sub_key]:
                    has_differences = True
                    break

        if not has_differences:
            return key

    # Create a new unique key for the hyper parameters
    for i in range(10000):
        key = f"{i:04}"
        if key not in previous_trainings:
            return key
    else:
        raise RuntimeError("Too many trained models")
