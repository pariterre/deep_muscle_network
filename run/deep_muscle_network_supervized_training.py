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

import biorbd
from deep_muscle_network import HyperParametersModel, PredictionModelMode
from deep_muscle_network.utils import FileIoHelpers


# TODO : setup logging for this file


def main(
    hyper_parameters_model: HyperParametersModel,
    prediction_model_mode: PredictionModelMode,
    biorbd_model: biorbd.Model,
    save_load_folder: str,
    force_retrain: bool,
    muscle_name,
    retrain,
    file_path,
    with_noise,
    plot_preparation,
    plot_loss_acc,
    plot_loader,
    save,
):
    """
    Main function to prepare, train, validate, test, and save a model.

    Parameters
    ----------
    hyper_parameters_model: HyperParametersModel
        HyperparametersModel, all hyperparameters chosen by the user.
    prediction_model_mode: PredictionModelMode
        Mode for the prediction model.
    biorbd_model: biorbd.Model
        The biorbd model that will be used to compute refence data.
    save_load_folder: str
        Path to the folder where the model will be loaded. If no model is found, then a new one will be trained and saved
        in this folder. This behaviour can be overriden by setting [force_retrain] to True.
    force_retrain: bool
        If True, the model will be retrained even if a model is found in the [save_load_folder]. If no model is found,
        then this parameter has no effect.



    - folder_name: str, path/name of the folder containing all CSV data files for muscles (one for each muscle).
    - muscle_name: str, name of the muscle.
    - retrain: bool, True to train the model again.
    - file_path: str, the path where the model will be saved after training.
    - with_noise: bool, (default = True), True to include noisy data in the dataset for learning.
    - plot_preparation: bool, True to show the distribution of all data preparation.
    - plot_loss_acc: bool, True to show plot loss, accuracy, predictions/targets.
    - plot_loader: bool, True to show plot comparing the loader data (train, validation, test) with the target data
        Warning : This plot is not available if the output height is set too large, as there may be too many subplots to display!
    - save: bool, True to save the model to file_path.
    """

    # Create a folder for save plots
    folder_name_muscle = f"{save_load_folder}/{muscle_name}"
    FileIoHelpers.mkdir(f"{folder_name_muscle}/_Model")  # Muscle/Model

    # Train_model if retrain == True or if none file_path already exist
    if retrain or os.path.exists(f"{folder_name_muscle}/_Model/{file_path}") == False:
        create_directory(f"{folder_name_muscle}/_Model/{file_path}")  # Muscle/Model

        # Prepare datas for trainning
        train_loader, val_loader, test_loader, input_size, output_size, y_labels = create_loaders_from_folder(
            hyper_parameters_model,
            prediction_model_mode,
            nb_q,
            nb_segment,
            num_datas_for_dataset=25000,
            folder_name=f"{folder_name_muscle}",
            muscle_name=muscle_name,
            with_noise=with_noise,
            plot=plot_preparation,
        )
        # Trainning
        model, _, _, _, _, _ = train_model_supervised_learning(
            train_loader,
            val_loader,
            test_loader,
            input_size,
            output_size,
            hyper_parameters_model,
            f"{folder_name_muscle}/_Model/{file_path}",
            plot_loss_acc,
            save,
            show_plot=True,
        )
        if plot_loader:
            # Visualize tranning : predictions/targets for loaders train, val and test
            visualize_prediction_training(
                model, f"{folder_name_muscle}/_Model/{file_path}", y_labels, train_loader, val_loader, test_loader
            )
    # Visualize : predictions/targets for all q variation
    visualize_prediction(
        prediction_model_mode,
        hyper_parameters_model.batch_size,
        nb_q,
        nb_segment,
        nb_muscle,
        f"{folder_name_muscle}/_Model/{file_path}",
        f"{folder_name_muscle}/plot_all_q_variation_",
    )


if __name__ == "__main__":
    main()
