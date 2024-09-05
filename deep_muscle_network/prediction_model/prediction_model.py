import logging

import numpy as np
import torch

from .data_set import DataSet
from .neural_network_utils.neural_network_model import NeuralNetworkModel
from .utils.prediction_model_folder_structure import PredictionModelFolderStructure


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

    def train(self, training_set: DataSet, validation_set: DataSet) -> None:
        """
        Train a model using supervised learning.

        Parameters
        ----------
        training_set : DataSet
            The training set.
        validation_set : DataSet
            The validation set.
        neural_network_model : NeuralNetworkModel
            The neural network model to use for the prediction model.


        Returns:
        - model (nn.Module): Trained model.
        - val_loss (float): Validation loss.
        - test_acc (float): Test accuracy (mean distance).
        - test_error (float): Test error.
        - test_abs_error (float): Test absolute error.
        - epoch (int): Number of epochs completed.
        """
        # TODO : Test this function
        _logger.info("Training the model...")

        # Set up optimizer and learning rate scheduler
        if not self._neural_network_model.is_initialized:
            self._neural_network_model.initialize(
                input_layer_node_count=training_set.input_len, output_layer_node_count=training_set.output_len
            )
        min_lr = 1e-8  # min lr could be attend with scheduler
        patience_scheduler = 20
        patience_early_stopping = 50  # Choose a early-stopping patience = 2 * scheduler patience

        # More details about scheduler in documentation
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._neural_network_model.optimizer, mode="min", factor=0.1, patience=patience_scheduler, min_lr=min_lr
        )
        # Initialization of EarlyStopping
        # More informations about Early stopping in doccumentation
        early_stopping = EarlyStopping(
            monitor="val_mae", patience=patience_early_stopping, min_delta=1e-9, verbose=True
        )

        for epoch in range(neural_network_model.num_epochs):
            train_loss, train_acc, train_error, train_abs_error = train(
                model, train_loader, neural_network_model.optimizer, neural_network_model.criterion
            )
            val_loss, val_acc, val_error, val_abs_error = evaluate(model, val_loader, neural_network_model.criterion)

            # Check for NaN values in accuracy
            # Sometimes, acc(s) could be Nan :(
            # Check your activation function and try an other !
            if np.isnan(train_acc) or np.isnan(val_acc):
                return model, float("inf"), float("inf")

            print(
                f"Epoch [{epoch+1}/{neural_network_model.num_epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f},",
                f"Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}, lr = {scheduler.get_last_lr()}",
            )

            #  if patience_scheduler, adjust/reduce learning rate
            scheduler.step(val_loss)

            # if patience_early_stopping, stop trainning to avoid overfitting
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping at epoch:", epoch + 1)
                break

        # TODO : Save the model
        return model, val_loss, test_acc, test_error, test_abs_error, epoch
