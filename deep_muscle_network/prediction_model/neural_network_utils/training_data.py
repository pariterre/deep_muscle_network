from dataclasses import dataclass, field
import json

from .neural_network_folder_structure import NeuralNetworkFolderStructure
from .data_set import DataSet


@dataclass
class TrainingData:
    training_data_set: DataSet
    validation_data_set: DataSet

    epoch_count: int = 0

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

    def save(self, base_folder: NeuralNetworkFolderStructure, model_file_name: str) -> None:
        # Save the training values
        training_values = {
            "epoch_count": self.epoch_count,
            "training_loss": self.training_loss,
            "training_accuracy": self.training_accuracy,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
        }
        with open(base_folder.training_values_file_path(model_name=model_file_name), "w") as file:
            json.dump(training_values, file, indent=2)

        # Save the training data set#
        # RENDU ICI!!!!!!!!
        self.training_data_set.save(save_path)
