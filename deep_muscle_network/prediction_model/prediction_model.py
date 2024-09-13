from copy import deepcopy
import logging

import numpy as np
import torch

from .data_set import DataSet
from .neural_network_utils.neural_network import NeuralNetwork
from .neural_network_utils.neural_network_folder_structure import NeuralNetworkFolderStructure
from .neural_network_utils.loss_methods import LossFunctionAbstract, LossFunctionConstructors
from .neural_network_utils.stopping_conditions import StoppingConditionMaxEpochs
from .neural_network_utils.torch_utils import get_torch_device
from ..reference_model.reference_model_abstract import ReferenceModelAbstract
from ..plotter.plotter_abstract import PlotterAbstract

_logger = logging.getLogger(__name__)


class PredictionModel:
    def __init__(self, path: str):
        """
        Initialize the prediction model.

        Parameters
        ----------
        path : str
            The base folder where the prediction model will be loaded and saved.
        """
        self._folder_structure = NeuralNetworkFolderStructure(path)
        self._neural_network: NeuralNetwork | None

    def _set_neural_network(
        self,
        reference_model: ReferenceModelAbstract,
        neural_network: NeuralNetwork,
    ) -> None:
        """
        Set the neural network configuration for the prediction model.

        Parameters
        ----------
        reference_model : ReferenceModelAbstract
            Reference model for the prediction model. It is used to create the training and validation datasets.
        neural_network : NeuralNetwork
            The neural network configuration to use to train the model.
        """

        self._neural_network = deepcopy(neural_network)

        # Put the output_scaling_vector to the right device
        self._neural_network.set_reference_values(
            input_layer_node_count=len(reference_model.input_labels),
            output_layer_node_count=len(reference_model.output_labels),
            output_scaling_vector=reference_model.scaling_vector,
        )

    def save(self, reference_model: ReferenceModelAbstract) -> None:
        """
        Save the model configuration to a file.

        Parameters
        ----------
        reference_model : ReferenceModelAbstract
            Reference model for the prediction model. It is used to create the training and validation
        """
        # TODO : Test this method
        if self._neural_network is None:
            raise ValueError(
                "No hyperparameters have been set for the prediction model, please train or load a model before saving."
            )

        # Save the scaling vector
        self._neural_network.save(base_folder=self._folder_structure, model_name=reference_model.name)

    def load_if_exists(
        self,
        reference_model: ReferenceModelAbstract,
        neural_network: NeuralNetwork,
        plotter: PlotterAbstract | None = None,
    ) -> None:
        """
        Load the model configuration from a file if it exists, otherwise it trains a new model.

        Parameters
        ----------
        reference_model : ReferenceModelAbstract
            Reference model for the prediction model. It is used to load the model if it exists. Otherwise, it is used to
            create the training and validation datasets.
        neural_network : NeuralNetwork
            The neural network configuration to use to train the model. Ignored if the model already exists.
        plotter : PlotterAbstract
            The plotter to use to visualize the training, validation, and test results. If None, no plot will be generated.
            Ignored if the model already exists.

        """
        # TODO : Test this method
        try:
            self.load(reference_model, neural_network)
        except:
            self.train(reference_model, neural_network, plotter)

    def load(
        self,
        reference_model: ReferenceModelAbstract,
        neural_network: NeuralNetwork,
    ) -> None:
        """
        Load the model configuration from a file.

        Parameters
        ----------
        reference_model : ReferenceModelAbstract
            Reference model for the prediction model. It is used to create the training and validation
        """

        # TODO : Test this method
        self._set_neural_network(reference_model, neural_network)
        self._neural_network.load(base_folder=self._folder_structure, model_name=reference_model.name)

    def train(
        self,
        reference_model: ReferenceModelAbstract,
        neural_network: NeuralNetwork,
        plotter: PlotterAbstract | None = None,
    ) -> None:
        """
        Train a model using supervised learning. The model is automatically saved after training.

        Parameters
        ----------
        reference_model : ReferenceModelAbstract
            Reference model for the prediction model. It is used to create the training and validation datasets.
        hyper_parameters : HyperParameters
            Hyperparameters used to train the model.
        plotter : PlotterAbstract
            The plotter to use to visualize the training, validation, and test results. If None, no plot will be generated.
        """

        # TODO : Test this function
        self._set_neural_network(reference_model, neural_network)

        _logger.info("Training the model...")
        training_data_set = reference_model.generate_dataset(data_point_count=self._neural_network.training_data_count)
        validation_data_set = reference_model.generate_dataset(
            data_point_count=self._neural_network.validation_data_count
        )

        # More details about scheduler in documentation
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._neural_network.optimizer, mode="min", factor=0.1, patience=20, min_lr=1e-8
        )

        # Training values which correspond to the training losses, training accuracies, validation losses, and validation accuracies
        training_values: list[tuple[float, float, float, float]] = []

        current_loss = torch.inf
        max_epochs = min(
            cond.max_epochs
            for cond in self._neural_network.stopping_conditions
            if isinstance(cond, StoppingConditionMaxEpochs)
        )
        while not any(
            [condition.should_stop(current_loss=current_loss) for condition in self._neural_network.stopping_conditions]
        ):
            # TODO Add shuffling of the data?

            training_loss, training_accuracy = self._perform_epoch_training(
                data_set=training_data_set, loss_function=self._neural_network.loss_function
            )
            validation_loss, validation_accuracy = self._perform_epoch_training(
                data_set=validation_data_set, loss_function=self._neural_network.loss_function, only_compute=True
            )

            # Sanity check, if the loss is NaN, the training failed, you can check your activation function
            if np.isnan(training_accuracy) or np.isnan(validation_accuracy):
                break

            epoch_count = len(training_values)
            _logger.info(
                f"Epoch [{epoch_count}/{max_epochs}]\n"
                f"\tLoss values: training={training_loss:.8f}, validation={validation_loss:.8f}\n"
                f"\tAccuracies: training={training_accuracy:.6f}, validation={validation_accuracy:.6f}\n"
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

        inputs = data_set.inputs.T
        out = self._neural_network.model(inputs)
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
        return output_vector * self._neural_network.output_scaling_vector

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
        return output_vector / self._neural_network.output_scaling_vector

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
            self._neural_network.model.train()

            # If it is training, we are updating the model with each prediction, we therefore need to do it in a loop
            running_loss = 0.0
            all_predictions = torch.tensor([]).to(get_torch_device())
            all_targets = torch.tensor([]).to(get_torch_device())
            for data in data_set:
                targets = self._normalize_output_vector(data.targets.T)
                self._neural_network.optimizer.zero_grad()

                # Get the predictions and targets
                outputs = self.predict(data, normalized=True).T

                # Do some machine learning shenanigans
                current_loss = loss_function.forward(outputs, targets)
                current_loss.backward()  # Backpropagation
                self._neural_network.optimizer.step()  # Updating weights

                # Populate the return values
                running_loss += current_loss
                all_predictions = torch.cat((all_predictions, outputs))
                all_targets = torch.cat((all_targets, targets))

            # Put back the model in evaluation mode
            self._neural_network.model.eval()

        # Calculation of average loss
        epoch_loss = (running_loss / len(data_set)).item()

        # Calculation of mean distance and error %
        epoch_accuracy = (all_predictions - all_targets).abs().mean().item()

        return epoch_loss, epoch_accuracy
