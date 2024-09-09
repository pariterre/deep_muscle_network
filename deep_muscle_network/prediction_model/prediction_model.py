import logging
import os

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
from ..plotter.plotter_abstract import PlotterAbstract

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

        self._scaling_vector = None

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

    def save(self, reference_model: ReferenceModelAbstract) -> None:
        """
        Save the model configuration to a file.

        Parameters
        ----------
        reference_model : ReferenceModelAbstract
            Reference model for the prediction model. It is used to create the training and validation
        """
        # TODO : Test this method
        self._neural_network_model.save(model_name=reference_model.name, folder=self._folder_structure)

        # Save the scaling vector
        folder = os.path.dirname(self._folder_structure.trained_model_path(reference_model.name))
        filepath = os.path.join(folder, "scaling_vector.npy")
        np.save(filepath, self._scaling_vector)

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

        # Load the scaling vector
        folder = os.path.dirname(self._folder_structure.trained_model_path(reference_model.name))
        filepath = os.path.join(folder, "scaling_vector.npy")
        self._scaling_vector = torch.tensor(np.load(filepath))

    def train(
        self,
        reference_model: ReferenceModelAbstract,
        number_data_points: tuple[int, int],
        loss_function: LossFunctionAbstract = LossFunctionConstructors.MODIFIED_HUBER(delta=0.2, factor=1.0),
        stopping_conditions: tuple[StoppingConditionsAbstract, ...] = (
            StoppingConditionConstructors.MAX_EPOCHS(max_epochs=1000),
            StoppingConditionConstructors.HAS_STOPPED_IMPROVING(patience=50, epsilon=1e-5),
        ),
        plotter: PlotterAbstract | None = None,
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
        plotter : PlotterAbstract
            The plotter to use to visualize the training, validation, and test results. If None, no plot will be generated.
        """
        # TODO : Test this function
        _logger.info("Training the model...")
        training_data_set = reference_model.generate_dataset(data_point_count=number_data_points[0])
        validation_data_set = reference_model.generate_dataset(data_point_count=number_data_points[1])
        self._scaling_vector = reference_model.scaling_vector

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

        # Training values which correspond to the training losses, training accuracies, validation losses, and validation accuracies
        training_values: list[tuple[float, float, float, float]] = []

        current_loss = torch.inf
        max_epochs = min(
            cond.max_epochs for cond in stopping_conditions if isinstance(cond, StoppingConditionMaxEpochs)
        )
        while not any([condition.should_stop(current_loss=current_loss) for condition in stopping_conditions]):
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

            epoch_count = len(training_values)
            _logger.info(
                f"Epoch [{epoch_count}/{max_epochs}]\n"
                f"\tLoss values: trainning={training_loss:.8f}, validation={validation_loss:.8f}\n"
                f"\tAccuracies: trainning={training_accuracy:.6f}, validation={validation_accuracy:.6f}\n"
                f"\tCurrent learning rate = {scheduler.get_last_lr()}"
            )

            scheduler.step(validation_loss)  # Adjust/reduce learning rate
            training_values.append((training_loss, validation_loss, training_accuracy, validation_accuracy))
            current_loss = validation_loss

        # Save and plot the results
        _logger.info(f"Training complete, final loss: {training_loss:.8f}")
        self.save(reference_model)
        if plotter is not None:
            plotter.plot_loss_and_accuracy(training_values)

    def predict(self, data_set: DataSet, normalized: bool = False) -> torch.Tensor:
        """
        Predict the output of the model for a given input.

        Parameters
        ----------
        data_set : DataSet
            The data set to predict.
        normalized : bool
            If True, the output will be normalized. If False, the output remains in the original scale.

        Returns
        -------
        torch.Tensor
            The predicted output.
        """
        # TODO : Test this function

        device = self._get_device()
        inputs = data_set.inputs.T.to(device)
        out = self._neural_network_model.prediction_model(inputs)
        if not normalized:
            out = self._denormalize_output_vector(out)
        return out.T

    def _normalize_output_vector(self, output_vector: torch.Tensor) -> torch.Tensor:
        """
        Normalize the output vector.

        Parameters
        ----------
        output_vector : torch.Tensor
            The output vector to normalize.

        Returns
        -------
        torch.Tensor
            The normalized output vector.
        """
        # TODO : Test this function
        return output_vector / self._scaling_vector

    def _denormalize_output_vector(self, output_vector: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the output vector.

        Parameters
        ----------
        output_vector : torch.Tensor
            The output vector to denormalize.

        Returns
        -------
        torch.Tensor
            The denormalized output vector.
        """
        # TODO : Test this function
        return output_vector * self._scaling_vector

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
            with torch.no_grad():
                all_predictions = self.predict(data_set, normalized=True).T
                all_targets = self._normalize_output_vector(data_set.targets.T)
                running_loss = loss_function.forward(all_predictions, all_targets).sum()

        else:
            # Put the model in training mode
            self._neural_network_model.prediction_model.train()

            # If it is trainning, we are updating the model with each prediction, we therefore need to do it in a loop
            running_loss = 0.0
            all_predictions = torch.tensor([])
            all_targets = torch.tensor([])
            for data in data_set:
                targets = self._normalize_output_vector(data.targets.T.to(self._get_device()))
                self._neural_network_model.optimizer.zero_grad()

                # Get the predictions and targets
                outputs = self.predict(data, normalized=True).T

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
        epoch_accuracy = (all_predictions - all_targets).abs().mean().item()

        return epoch_loss, epoch_accuracy
