from copy import deepcopy
from itertools import product
import logging

from deep_muscle_network import (
    PredictionModel,
    ReferenceModel,
    ReferenceModelBiorbd,
    PredictionModelUtils,
    BiorbdOutputModes,
    NeuralNetwork,
    StoppingConditionConstructors,
    LossFunctionConstructors,
    TrainingData,
    Plotter,
    PlotterMatplotlib,
)
import numpy as np


def _generate_configurations(
    training_data_count: tuple[int],
    validation_data_count: tuple[int],
    hidden_layers_node_count: tuple[tuple[int]],
    use_batch_norm: tuple[bool],
    stopping_conditions: tuple[tuple[StoppingConditionConstructors]],
    loss_function: tuple[LossFunctionConstructors],
    learning_rate: tuple[float],
    dropout_probability: tuple[float],
) -> list[NeuralNetwork]:
    """
    Generate all the configurations to train.

    Returns
    -------
    list[NeuralNetwork]
        The list of all the permutations of the configurations.
    """

    return [
        NeuralNetwork(
            training_data_count=configuration[0],
            validation_data_count=configuration[1],
            hidden_layers_node_count=configuration[2],
            use_batch_norm=configuration[3],
            stopping_conditions=configuration[4],
            loss_function=configuration[5],
            learning_rate=configuration[6],
            dropout_probability=configuration[7],
        )
        for configuration in product(
            training_data_count,
            validation_data_count,
            hidden_layers_node_count,
            use_batch_norm,
            stopping_conditions,
            loss_function,
            learning_rate,
            dropout_probability,
        )
    ]


def main(
    prediction_model: PredictionModel,
    reference_model: ReferenceModel,
    neural_networks: list[NeuralNetwork],
    force_retrain: bool,
    plotter: Plotter,
) -> None:
    """
    Train a model with multiple hyperparameters and print the best hyperparameters.

    Parameters
    ----------
    prediction_model: PredictionModel
        The prediction model to train.
    reference_model: ReferenceModelAbstract
        Reference model for the prediction model. It is used to create the training, validation, and test datasets.
    force_retrain: bool
        If True, the model will be retrained even if a model is found in the [save_and_load_folder]. If no model is found,
        then this parameter has no effect.
    """

    # Before beggining, compute an estimation of execution time
    # The user can choose to stop if the execution is to long according to him
    # For example, if estimed execution time if around 100 hours... maybe you have to many hyperparameters to try ...
    logging.info(f"------------------------")
    PredictionModelUtils.print_estimated_training_time(len(neural_networks))
    logging.info(f"------------------------")

    # Train or load the models for each configuration
    prediction_models: list[PredictionModel] = []
    training_values: list[TrainingData] = []
    for index, neural_network in enumerate(neural_networks):
        logging.info(f"Training model {index + 1}/{len(neural_networks)}")

        # Reset the numpy seed to ensure the same data set is generated for each model
        np.random.seed(42)
        training_values.append(
            prediction_model.train(reference_model=reference_model, neural_network=neural_network)
            if force_retrain
            else prediction_model.load_if_exists(reference_model=reference_model, neural_network=neural_network)
        )
        prediction_models.append(deepcopy(prediction_model))

    # Compute the prediction time for each model
    logging.info("Computing prediction time for each model")
    np.random.seed(42)
    prediction_time_data_set = reference_model.generate_dataset(data_point_count=1000)
    prediction_times: list[int] = [
        PredictionModelUtils.compute_prediction_time(
            prediction_model=prediction_model, data_set=prediction_time_data_set
        )
        for prediction_model in prediction_models
    ]

    # Find the model with the lowest final loss values
    best_index: int = 0
    for index, training_value in enumerate(training_values):
        if training_value.validation_loss[-1] < training_values[best_index].validation_loss[-1]:
            best_index = index

    # Print the best hyper parameters
    logging.info(f"Best hyper parameters loss: {training_values[best_index].validation_loss[-1]}\n")

    # Plot visualisation to compare all model trained (pareto front)
    plotter.pareto_front(
        x_data=np.array(prediction_times),
        y_data=np.array([training_value.validation_loss[-1] for training_value in training_values]),
        title="Pareto front (Validation loss vs prediction time)",
        x_label="Execution time (s)",
        y_label="Validation loss",
    )
    plotter.pareto_front(
        x_data=np.array(prediction_times),
        y_data=np.array([training_value.training_time for training_value in training_values]),
        title="Pareto front (Validation loss vs training time)",
        x_label="Execution time (s)",
        y_label="Validation loss",
    )

    # Finally, plot figures for the best model
    prediction_model = prediction_models[best_index]
    neural_network = neural_networks[best_index]
    prediction_model.load(reference_model=reference_model, neural_network=neural_network, plotter=plotter)

    # Vizualize the prediction quality
    test_data_set = reference_model.generate_dataset(data_point_count=250)
    test_data_set.fill_predictions(prediction_model, reference_model)
    plotter.plot_prediction(test_data_set)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(
        prediction_model=PredictionModel(path="TrainedModels"),
        reference_model=ReferenceModelBiorbd(
            biorbd_model_path="models/Wu_DeGroote.bioMod",
            muscle_names=("PECM2", "PECM3"),
            output_mode=BiorbdOutputModes.MUSCLE,
            muscle_tendon_length_normalization=1.0 * 1000.0,
        ),
        neural_networks=_generate_configurations(
            training_data_count=(2500,),
            validation_data_count=(250,),
            hidden_layers_node_count=((128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)),
            use_batch_norm=(True,),
            stopping_conditions=(
                (
                    StoppingConditionConstructors.MAX_EPOCHS(max_epochs=1),
                    StoppingConditionConstructors.HAS_STOPPED_IMPROVING(patience=50, epsilon=1e-5),
                ),
            ),
            loss_function=(LossFunctionConstructors.MODIFIED_HUBER(delta=0.2, factor=1.0),),
            learning_rate=(1e-2,),
            dropout_probability=(0.0, 0.2),
        ),
        force_retrain=False,
        plotter=PlotterMatplotlib(),
    )
