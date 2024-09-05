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

    @property
    def has_a_trained_model(self) -> bool:
        # TODO : Test this function
        return self._folder_structure.has_a_trained_model

    def train(
        self,
        training_data_set: DataSet,
        validation_data_set: DataSet,
        loss_function: LossFunctionAbstract = LossFunctionConstructors.MODIFIED_HUBER(delta=0.2, factor=1.0),
        stopping_conditions: tuple[StoppingConditionsAbstract, ...] = (
            StoppingConditionConstructors.MAX_EPOCHS(max_epochs=1000),
            StoppingConditionConstructors.HAS_STOPPED_IMPROVING(patience=50, epsilon=1e-5),
        ),
    ) -> float:
        """
        Train a model using supervised learning.

        Parameters
        ----------
        training_set : DataSet
            The training set.
        validation_set : DataSet
            The validation set.
        loss_function : LossFunctionAbstract
            The loss function to use during training.
        stopping_conditions : tuple[StoppingConditionsAbstract, ...]
            The stopping conditions to use during training.

        Returns
        -------
        float
            The final training loss.
        """
        # TODO : Test this function
        _logger.info("Training the model...")

        # Set up optimizer and learning rate scheduler
        if not self._neural_network_model.is_initialized:
            self._neural_network_model.initialize(
                input_layer_node_count=training_data_set.input_len, output_layer_node_count=training_data_set.output_len
            )

        # More details about scheduler in documentation
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._neural_network_model.optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-8
        )

        current_loss = torch.inf
        epoch_index = 0
        max_epochs = min(
            cond.max_epochs for cond in stopping_conditions if isinstance(cond, StoppingConditionMaxEpochs)
        )
        while not any([condition.should_stop(current_loss=current_loss) for condition in stopping_conditions]):
            training_loss, training_accuracy = self._perform_predictions(
                data_set=training_data_set, loss_function=loss_function, is_training=True
            )
            validation_loss, validation_accuracy = self._perform_predictions(
                data_set=validation_data_set, loss_function=loss_function
            )

            # Check for NaN values in accuracy
            # Sometimes, acc(s) could be Nan :(
            # Check your activation function and try an other !
            if np.isnan(training_accuracy) or np.isnan(validation_accuracy):
                break

            _logger.info(
                f"Epoch [{epoch_index}/{max_epochs}]\n"
                f"\tLoss values: trainning={training_loss:.8f}, validation={validation_loss:.8f}\n",
                f"\tAccuracies: trainning={training_accuracy:.6f}, validation={validation_accuracy:.6f}\n"
                f"\tCurrent learning rate = {scheduler.get_last_lr()}",
            )

            scheduler.step(validation_loss)  # Adjust/reduce learning rate
            current_loss = training_loss
            epoch_index += 1

        # TODO : Save the model
        return training_loss

    def _perform_predictions(
        self,
        data_set: DataSet,
        loss_function: LossFunctionAbstract = LossFunctionConstructors.MODIFIED_HUBER(delta=0.2, factor=1.0),
        is_training: bool = False,
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

        Returns
        -------
        tuple[float, float]
            The loss and the accuracy of the predictions.
        """

        # Get some aliases
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prediction_model = self._neural_network_model.prediction_model
        # Put the model in training mode
        if is_training:
            prediction_model.train()
            self._neural_network_model.optimizer.zero_grad()
        else:
            prediction_model.eval()

        # Perform the training
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        for inputs, targets in data_set:
            # Get the predictions and targets
            inputs: torch.Tensor
            targets: torch.Tensor
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = prediction_model(inputs)
            current_loss = loss_function.forward(outputs, targets)

            # Do some machine learning shenanigans
            if is_training:
                current_loss.backward()  # Backpropagation
                self._neural_network_model.optimizer.step()  # Updating weights

            # Stats for plot
            running_loss += current_loss.item() * inputs.size(0)
            all_predictions.append(outputs)
            all_targets.append(targets)

        # Calculation of average loss
        epoch_loss = running_loss / len(data_set)

        # Calculation of mean distance and error %
        epoch_accuracy = (torch.tensor(all_predictions) - torch.tensor(all_targets)).mean().item()

        return epoch_loss, epoch_accuracy
