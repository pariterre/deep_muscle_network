import os


class NeuralNetworkFolderStructure:
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
    def model_folder(self) -> str:
        # TODO : Test this function
        return os.path.join(self._base_folder, "Models")

    def trained_model_path(self, model_name: str) -> str:
        # TODO : Test this function
        return os.path.join(self.model_folder, f"{model_name}.pth")

    @property
    def hyper_parameters_model_path(self) -> str:
        # TODO : Test this function
        return os.path.join(self.model_folder, f"hyper_parameters.json")

    @property
    def data_folder(self) -> str:
        # TODO : Test this function
        return os.path.join(self._base_folder, "Data")

    @property
    def training_data_folder(self) -> str:
        # TODO : Test this function
        return os.path.join(self.data_folder, "Training")

    def training_values_file_path(self, model_name: str) -> str:
        # TODO : Test this function
        return os.path.join(self.training_data_folder, f"{model_name}.json")

    def training_data_set_file_path(self, model_name: str) -> str:
        # TODO : Test this function
        return os.path.join(self.training_data_folder, f"{model_name}_training_data_set.pth")

    def validation_data_set_file_path(self, model_name: str) -> str:
        # TODO : Test this function
        return os.path.join(self.training_data_folder, f"{model_name}_validation_data_set.pth")

    @property
    def results_folder(self) -> str:
        # TODO : Test this function
        return os.path.join(self._base_folder, "Results")

    def _create_folder_structure(self):
        # TODO : Test this function
        _mkdir_if_not_exist(self._base_folder)
        _mkdir_if_not_exist(self.model_folder)
        _mkdir_if_not_exist(self.data_folder)
        _mkdir_if_not_exist(self.training_data_folder)
        _mkdir_if_not_exist(self.results_folder)


def _mkdir_if_not_exist(folder_path: str):
    """
    Create a new directory if it does not already exist. Otherwise, do nothing.

    Parameters
    ----------
    folder_path : str
        The path to the directory to create.
    """
    # TODO : Test this function
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
