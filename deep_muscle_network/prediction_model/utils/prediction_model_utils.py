import logging
from time import time

from ..neural_network_utils.data_set import DataSet
from ..prediction_model import PredictionModel


class PredictionModelUtils:
    @staticmethod
    def print_estimated_training_time(
        configuration_count: int, estimated_training_time_per_configuration: int = 300
    ) -> None:
        """
        Compute an estimation of the time needed to train all configurations and print it. That is a fairly rough estimation
        and should be taken with a grain of salt, but it can be useful to get an idea of the order of magnitude of the time
        needed.

        Parameters
        ----------
        configuration_count : int
            The number of configurations to train.
        estimated_training_time_per_configuration : int
            The estimated time in seconds needed to train one configuration. Default is 5 minutes per configuration.

        Returns
        -------
        None
            Print the estimated time needed to train all configurations.
        """

        # Estimate the total time for all combinations in seconds
        total_time = configuration_count * estimated_training_time_per_configuration

        # Print the estimated time needed to train all configurations on the formating XXhYY
        logging.info(
            f"Estimated time to train all ({configuration_count}) configurations: {total_time // 3600}h{total_time % 3600 // 60:02}"
        )

    @staticmethod
    def compute_prediction_time(prediction_model: PredictionModel, data_set: DataSet):
        """
        Compute the average time taken to predict output. This function can be used, for instance, to compare the
        performance of different models on a given data set.

        Parameters
        ----------
        prediction_model : PredictionModel
            The prediction model to evaluate.
        data_set : DataSet
            The data set to use to evaluate the prediction time.
        """

        tic = time()
        prediction_model.predict(data_set=data_set)
        toc = time()

        # Return the mean execution time in seconds
        return (toc - tic) / len(data_set)
