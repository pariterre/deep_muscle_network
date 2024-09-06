import logging

import numpy as np
import torch

from .data_set import DataSet
from .neural_network_utils.neural_network_model import NeuralNetworkModel
from .utils.prediction_model_folder_structure import PredictionModelFolderStructure
from .neural_network_utils.loss_methods import LossFunctionAbstract, LossFunctionConstructors
from .neural_network_utils.stopping_conditions import (
    StoppingConditionsAbstract,
    StoppingConditionConstructors,
    StoppingConditionMaxEpochs,
)
from ..reference_model.reference_model_abstract import ReferenceModelAbstract

_logger = logging.getLogger(__name__)


class PredictionModel:
    def __init__(
        self,
        path: str,
        neural_network_model: NeuralNetworkModel,
    ):
        """
        Initialize the prediction model.

        Parameters
        ----------
        path : str
            The base folder where the prediction model will be loaded and saved.
        neural_network_model : NeuralNetworkModel
            The neural network model to use for the prediction model.
        """
        self._folder_structure = PredictionModelFolderStructure(path)
        self._neural_network_model = neural_network_model

    def has_a_trained_model(self, reference_model: ReferenceModelAbstract) -> bool:
        """
        Check if a trained model exists for the given reference model.

        Parameters
        ----------
        reference_model : ReferenceModelAbstract
            Reference model for the prediction model. It is used to create the training and validation datasets.

        Returns
        -------
        bool
            True if a trained model exists, False otherwise.
        """
        # TODO : Test this function
        return self._folder_structure.has_a_trained_model(reference_model.name)

    def load(self, reference_model: ReferenceModelAbstract) -> None:
        """
        Load the model configuration from a file.

        Parameters
        ----------
        reference_model : ReferenceModelAbstract
            Reference model for the prediction model. It is used to create the training and validation
        """

        # TODO : Test this method
        if not self._neural_network_model.is_initialized:
            self._neural_network_model.initialize(
                input_layer_node_count=len(reference_model.input_labels),
                output_layer_node_count=len(reference_model.output_labels),
            )

        self._neural_network_model.load(model_name=reference_model.name, folder=self._folder_structure)

    def train(
        self,
        reference_model: ReferenceModelAbstract,
        number_data_points: tuple[int, int],
        loss_function: LossFunctionAbstract = LossFunctionConstructors.MODIFIED_HUBER(delta=0.2, factor=1.0),
        stopping_conditions: tuple[StoppingConditionsAbstract, ...] = (
            StoppingConditionConstructors.MAX_EPOCHS(max_epochs=1000),
            StoppingConditionConstructors.HAS_STOPPED_IMPROVING(patience=50, epsilon=1e-5),
        ),
    ) -> None:
        """
        Train a model using supervised learning. The model is automatically saved after training.

        Parameters
        ----------
        reference_model : ReferenceModelAbstract
            Reference model for the prediction model. It is used to create the training and validation datasets.
        number_data_points : tuple[int, int]
            Number of data points to generate for the training and validation datasets, respectively.
        loss_function : LossFunctionAbstract
            The loss function to use during training.
        stopping_conditions : tuple[StoppingConditionsAbstract, ...]
            The stopping conditions to use during training.
        """
        # TODO : Test this function
        _logger.info("Training the model...")
        training_data_set = reference_model.generate_dataset(data_point_count=number_data_points[0])
        validation_data_set = reference_model.generate_dataset(data_point_count=number_data_points[1])

        # Set up optimizer and learning rate scheduler
        if not self._neural_network_model.is_initialized:
            self._neural_network_model.initialize(
                input_layer_node_count=len(reference_model.input_labels),
                output_layer_node_count=len(reference_model.output_labels),
            )

        # More details about scheduler in documentation
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._neural_network_model.optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-8
        )

        training_loss = torch.inf
        epoch_index = 0
        max_epochs = min(
            cond.max_epochs for cond in stopping_conditions if isinstance(cond, StoppingConditionMaxEpochs)
        )
        while not any([condition.should_stop(current_loss=training_loss) for condition in stopping_conditions]):
            # TODO Add shuffling of the data?

            training_loss, training_accuracy = self._perform_epoch_training(
                data_set=training_data_set, loss_function=loss_function
            )
            validation_loss, validation_accuracy = self._perform_epoch_training(
                data_set=validation_data_set, loss_function=loss_function, only_compute=True
            )

            # Sanity check, if the loss is NaN, the training failed, you can check your activation function
            if np.isnan(training_accuracy) or np.isnan(validation_accuracy):
                break

            _logger.info(
                f"Epoch [{epoch_index}/{max_epochs}]\n"
                f"\tLoss values: trainning={training_loss:.8f}, validation={validation_loss:.8f}\n"
                f"\tAccuracies: trainning={training_accuracy:.6f}, validation={validation_accuracy:.6f}\n"
                f"\tCurrent learning rate = {scheduler.get_last_lr()}"
            )

            scheduler.step(validation_loss)  # Adjust/reduce learning rate
            epoch_index += 1

        # Save the model
        _logger.info(f"Training complete, final loss: {training_loss:.8f}")
        self._neural_network_model.save(model_name=reference_model.name, folder=self._folder_structure)

    def predict(self, data_set: DataSet) -> torch.Tensor:
        """
        Predict the output of the model for a given input.

        Parameters
        ----------
        data_set : DataSet
            The data set to predict.

        Returns
        -------
        torch.Tensor
            The predicted output.
        """
        # TODO : Test this function

        device = self._get_device()
        inputs = data_set.inputs.to(device)
        return self._neural_network_model.prediction_model(inputs)

    def _get_device(self) -> torch.device:
        """
        Get the device to use for the computations.

        Returns
        -------
        torch.device
            The device to use.
        """
        # TODO : Test this function
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _perform_epoch_training(
        self,
        data_set: DataSet,
        loss_function: LossFunctionAbstract = LossFunctionConstructors.MODIFIED_HUBER(delta=0.2, factor=1.0),
        only_compute: bool = False,
    ) -> tuple[float, float]:
        """
        Perform all the predictions in the dataset and compute the associate loss function. If CUDA is available, the
        computations will automatically be performed on the GPU. If [is_training] is True, the model will be put in
        training mode, i.e. the gradients will be computed and the weights will be updated.

        Parameters
        ----------
        data_set : DataSet
            The dataset to perform the predictions on.
        loss_function : LossFunctionAbstract
            The loss function to use during training.
        only_compute : bool
            If True, the model will be put in evaluation mode, i.e. the gradients will not be computed and the weights

        Returns
        -------
        tuple[float, float]
            The loss and the accuracy of the predictions.
        """
        # TODO : Test this function

        # Perform the predictions
        if only_compute:
            all_predictions = self.predict(data_set)
            all_targets = data_set.outputs
            running_loss = loss_function.forward(all_predictions, all_targets).sum()

        else:
            self._neural_network_model.prediction_model.train()
            self._neural_network_model.optimizer.zero_grad()

            # If it is trainning, we are updating the model with each prediction, we therefore need to do it in a loop
            running_loss = 0.0
            all_predictions = torch.tensor([])
            all_targets = torch.tensor([])
            for data in data_set:
                # Get the predictions and targets
                targets: torch.Tensor = data.outputs
                outputs = self.predict(data)

                # Do some machine learning shenanigans
                current_loss = loss_function.forward(outputs, targets)
                current_loss.backward()  # Backpropagation
                self._neural_network_model.optimizer.step()  # Updating weights

                # Populate the return values
                running_loss += current_loss
                all_predictions = torch.cat((all_predictions, outputs))
                all_targets = torch.cat((all_targets, targets))

            # Put back the model in evaluation mode
            self._neural_network_model.prediction_model.eval()

        # Calculation of average loss
        epoch_loss = (running_loss / len(data_set)).item()

        # Calculation of mean distance and error %
        epoch_accuracy = (all_predictions - all_targets).mean().item()

        return epoch_loss, epoch_accuracy
