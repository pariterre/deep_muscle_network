from dataclasses import dataclass, field
import json
from typing import Any, Self

from .neural_network import NeuralNetwork
from .neural_network_folder_structure import NeuralNetworkFolderStructure
from .data_set import DataSet
from ...reference_model.reference_model import ReferenceModel


@dataclass
class TrainingData:
    training_data_set: DataSet
    validation_data_set: DataSet

    training_data_set_seed: Any = field(default=None)
    validation_data_set_seed: Any = field(default=None)

    epoch_count: int = field(init=False, default=0)
    training_time: float = field(init=False, default=None)

    training_loss: list[float] = field(init=False, default_factory=list)
    training_accuracy: list[float] = field(init=False, default_factory=list)

    validation_loss: list[float] = field(init=False, default_factory=list)
    validation_accuracy: list[float] = field(init=False, default_factory=list)

    def add_epoch(
        self, training_loss: float, training_accuracy: float, validation_loss: float, validation_accuracy: float
    ) -> None:
        """
        Add an epoch to the training data.

        Parameters
        ----------
        training_loss: float
            The training loss.
        training_accuracy: float
            The training accuracy.
        validation_loss: float
            The validation loss.
        validation_accuracy: float
            The validation accuracy

        """
        self.epoch_count += 1

        self.training_loss.append(training_loss)
        self.training_accuracy.append(training_accuracy)

        self.validation_loss.append(validation_loss)
        self.validation_accuracy.append(validation_accuracy)

    def set_training_time(self, training_time: float) -> None:
        """
        Set the training time.

        Parameters
        ----------
        training_time: float
            The training time in seconds.
        """
        object.__setattr__(self, "training_time", training_time)

    def save(self, base_folder: NeuralNetworkFolderStructure, model_file_name: str) -> None:
        # Save the training values
        training_values = {
            "training_data_set_seed": self.training_data_set_seed,
            "validation_data_set_seed": self.validation_data_set_seed,
            "epoch_count": self.epoch_count,
            "training_time": self.training_time,
            "training_loss": self.training_loss,
            "training_accuracy": self.training_accuracy,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
        }
        with open(base_folder.training_values_file_path(model_name=model_file_name), "w") as file:
            json.dump(training_values, file, indent=2)

    @classmethod
    def load(
        cls,
        neural_network: NeuralNetwork,
        reference_model: ReferenceModel,
        base_folder: NeuralNetworkFolderStructure,
        model_file_name: str,
    ) -> Self:
        with open(base_folder.training_values_file_path(model_name=model_file_name), "r") as file:
            training_values = json.load(file)

        training_data_set_seed = training_values["training_data_set_seed"]
        training_data_set = reference_model.generate_dataset(
            data_point_count=neural_network.training_data_count, seed=training_data_set_seed
        )

        validation_data_set_seed = training_values["validation_data_set_seed"]
        validation_data_set = reference_model.generate_dataset(
            data_point_count=neural_network.validation_data_count, seed=validation_data_set_seed
        )

        data = cls(
            training_data_set=training_data_set,
            training_data_set_seed=training_data_set_seed,
            validation_data_set=validation_data_set,
            validation_data_set_seed=validation_data_set_seed,
        )

        object.__setattr__(data, "epoch_count", training_values["epoch_count"])
        object.__setattr__(data, "training_loss", training_values["training_loss"])
        object.__setattr__(data, "training_time", training_values["training_time"])
        object.__setattr__(data, "training_accuracy", training_values["training_accuracy"])
        object.__setattr__(data, "validation_loss", training_values["validation_loss"])
        object.__setattr__(data, "validation_accuracy", training_values["validation_accuracy"])

        return data
