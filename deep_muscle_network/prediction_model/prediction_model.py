from .prediction_model_folder_structure import PredictionModelFolderStructure
from .prediction_model_output_modes import PredictionModelOutputModes


class PredictionModel:
    def __init__(
        self,
        path: str,
        output_mode: PredictionModelOutputModes,
    ):
        """
        Initialize the prediction model.

        Parameters
        ----------
        path : str
            The base folder where the prediction model will be loaded and saved.
        output_mode : PredictionModelOutputModes
            The mode for the prediction model.
        """
        self._output_mode = output_mode
        self._folder_structure = PredictionModelFolderStructure(path)

    @property
    def has_a_trained_model(self) -> bool:
        # TODO : Test this function
        return self._folder_structure.has_a_trained_model(self._output_mode)
