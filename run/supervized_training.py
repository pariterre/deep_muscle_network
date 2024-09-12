import logging

from deep_muscle_network import (
    ReferenceModelAbstract,
    PredictionModel,
    NeuralNetworkModel,
    ActivationMethodConstructors,
    StoppingConditionConstructors,
    ReferenceModelBiorbd,
    BiorbdOutputModes,
    PlotterAbstract,
    PlotterMatplotlib,
)


def main(
    prediction_model: PredictionModel,
    reference_model: ReferenceModelAbstract,
    number_training_data_points: tuple[int, int],
    force_retrain: bool,
    plotter: PlotterAbstract,
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
    number_training_data_points: tuple[int, int]
        Number of training and validation data points to use for training the model.
    force_retrain: bool
        If True, the model will be retrained even if a model is found in the [save_and_load_folder]. If no model is found,
        then this parameter has no effect.
    plotter: PlotterAbstract
        The plotter to use to visualize the training, validation, and test results.
    """

    # Create a folder for save plots
    # Train_model if retrain == True or if none file_path already exist
    if not prediction_model.has_a_trained_model(reference_model=reference_model) or force_retrain:
        prediction_model.train(
            reference_model=reference_model,
            number_data_points=number_training_data_points,
            stopping_conditions=(
                StoppingConditionConstructors.MAX_EPOCHS(max_epochs=1000),
                StoppingConditionConstructors.HAS_STOPPED_IMPROVING(patience=50, epsilon=1e-5),
            ),
            plotter=plotter,
        )
    else:
        prediction_model.load(reference_model=reference_model)

    test_data_set = reference_model.generate_dataset(data_point_count=250)
    test_data_set.fill_predictions(prediction_model, reference_model)

    # Vizualize the prediction quality
    plotter.plot_prediction(test_data_set)


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
        number_training_data_points=(2500, 250),
        force_retrain=False,
        plotter=PlotterMatplotlib(),
    )
