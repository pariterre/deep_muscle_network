# from neural_networks.data_preparation import create_loaders_from_folder, dataloader_to_tensor
# from neural_networks.data_tranning import train_model_supervised_learning
# from neural_networks.activation_functions import *
# import os
# from neural_networks.save_model import measure_time, save_model, main_function_model, del_saved_model
# from neural_networks.plot_visualisation import (
#     visualize_prediction_training,
#     visualize_prediction,
#     mean_distance,
#     compute_pourcentage_error,
# )
# from itertools import product
# from neural_networks.Loss import *
# from neural_networks.ModelHyperparameters import ModelHyperparameters
# from neural_networks.file_directory_operations import (
#     create_directory,
#     save_text_to_file,
#     save_informations_model,
#     get_min_value_from_csv,
# )
# import time
# from neural_networks.plot_pareto_front import plot_results_try_hyperparams
# import numpy as np
# from neural_networks.CSVBatchWriterTestHyperparams import CSVBatchWriterTestHyperparams

import logging

from deep_muscle_network import (
    ReferenceModelAbstract,
    PredictionModel,
    NeuralNetworkModel,
    ActivationMethodConstructors,
    ReferenceModelBiorbd,
    BiorbdOutputModes,
)


def main(
    prediction_model: PredictionModel,
    reference_model: ReferenceModelAbstract,
    force_retrain: bool,
):
    """
    Main function to prepare, train, validate, test, and save a model.

    Parameters
    ----------
    prediction_model: PredictionModel
        The prediction model to train.
    reference_model: ReferenceModelAbstract
        Reference model for the prediction model. It is used to create the training, validation, and test datasets.
    neural_network_model: NeuralNetworkModel
        The neural network model to use for the prediction model.
    force_retrain: bool
        If True, the model will be retrained even if a model is found in the [save_and_load_folder]. If no model is found,
        then this parameter has no effect.
    """

    # Create a folder for save plots
    # Train_model if retrain == True or if none file_path already exist
    if not prediction_model.has_a_trained_model or force_retrain:
        # Get the data sets
        training_data_set = reference_model.generate_dataset(data_point_count=250)
        validation_data_set = reference_model.generate_dataset(data_point_count=25)
        test_data_set = reference_model.generate_dataset(data_point_count=25)

        # Trainning
        model, _, _, _, _, _ = prediction_model.train(
            training_set=training_data_set, validation_set=validation_data_set
        )
        if plot_loader:
            # Visualize tranning : predictions/targets for loaders train, val and test
            visualize_prediction_training(
                model, f"{folder_name_muscle}/_Model/{file_path}", y_labels, train_loader, val_loader, test_loader
            )

    # Visualize : predictions/targets for all q variation
    visualize_prediction(
        prediction_model_output_mode,
        neural_network_model.batch_size,
        nb_q,
        nb_segment,
        nb_muscle,
        f"{folder_name_muscle}/_Model/{file_path}",
        f"{folder_name_muscle}/plot_all_q_variation_",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(
        prediction_model=PredictionModel(
            path="TrainedModels",
            neural_network_model=NeuralNetworkModel.from_default(
                "default",
                hidden_layers_node_count=(32, 32),
                activations=(ActivationMethodConstructors.GELU(), ActivationMethodConstructors.GELU()),
            ),
        ),
        reference_model=ReferenceModelBiorbd(
            biorbd_model_path="models/Wu_DeGroote.bioMod",
            muscle_names=("PECM2", "PECM3"),
            output_mode=BiorbdOutputModes.MUSCLE,
        ),
        force_retrain=True,
    )
