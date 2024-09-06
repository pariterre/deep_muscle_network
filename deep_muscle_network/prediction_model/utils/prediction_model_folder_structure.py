import os

from ...utils.file_io_helpers import FileIoHelpers


class PredictionModelFolderStructure:
    def __init__(self, base_folder: str) -> None:
        """
        Initialize the folder structure for the prediction model.

        Parameters
        ----------
        base_folder : str
            The base folder where the prediction model will be loaded and saved.
        """
        self._base_folder = base_folder

        # Create the full folder structure if it does not already exist
        self._create_folder_structure()

    @property
    def base_folder(self) -> str:
        # TODO : Test this function
        return self._base_folder

    @property
    def trained_model_folder(self) -> str:
        # TODO : Test this function
        return os.path.join(self._base_folder, "Models")

    @property
    def prediction_model_output_mode_path(self) -> str:
        # TODO : Test this function
        return os.path.join(self.trained_model_folder, f"prediction_model_output_mode.json")

    def trained_model_path(self, model_name: str) -> str:
        # TODO : Test this function
        return os.path.join(self.trained_model_folder, f"{model_name}.pth")

    def has_a_trained_model(self, model_name: str) -> bool:
        # TODO : Test this function
        return os.path.exists(self.trained_model_path(model_name))

    @property
    def data_set_folder(self) -> str:
        # TODO : Test this function
        return os.path.join(self._base_folder, "Data")

    @property
    def results_folder(self) -> str:
        # TODO : Test this function
        return os.path.join(self._base_folder, "Results")

    def _create_folder_structure(self):
        # TODO : Test this function
        FileIoHelpers.mkdir_if_not_exist(self._base_folder)
        FileIoHelpers.mkdir_if_not_exist(self.trained_model_folder)
        FileIoHelpers.mkdir_if_not_exist(self.data_set_folder)
        FileIoHelpers.mkdir_if_not_exist(self.results_folder)
