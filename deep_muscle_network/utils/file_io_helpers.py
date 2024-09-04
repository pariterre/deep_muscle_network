import logging
import os


class FileIoHelpers:
    @staticmethod
    def mkdir_if_not_exist(folder_path: str):
        """
        Create a new directory if it does not already exist. Otherwise, do nothing.

        Parameters
        ----------
        folder_path : str
            The path to the directory to create.
        """
        # TODO : Test this function
        if not os.path.exists(folder_path):
            logging.debug(f"The folder'{folder_path}' have been created.")
            os.makedirs(folder_path)
        else:
            logging.debug(f"The folder '{folder_path}' already exist.")


# def create_and_save_plot(directory_path, file_name):
#     """
#     Save a Matplotlib figure in a specific directory.

#     Args:
#     - directory_path (str): Path where the figure will be saved.
#     - file_name (str): Name of the figure file.
#     """

#     # Create the directory if it does not exist
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)

#     # Construct the full file path for the figure and save
#     file_path = os.path.join(directory_path, file_name)
#     plt.savefig(file_path)


# def save_text_to_file(text, file_path):
#     """
#     Save the given text to a file.

#     Args:
#         text (str): The text content to be saved.
#         file_path (str): The path to the file where the text will be saved.
#     """
#     # Open the file in write mode and save the text
#     with open(file_path, "w") as file:
#         file.write(text)


# def save_informations_model(
#     file_path,
#     num_try,
#     val_loss,
#     test_acc,
#     test_error,
#     test_abs_error,
#     train_timer,
#     mean_model_load_timer,
#     mean_model_timer,
#     try_hyperparams,
#     mode,
#     epoch,
#     criterion_name,
#     criterion_params,
# ):
#     """
#     Save model information and hyperparameters to a text file.

#     This function compiles various metrics, hyperparameters, and details about the model training process
#     and saves them in a formatted text file.

#     Args:
#         file_path (str): The directory path where the information text file will be saved.
#         num_try (int): Number of the model try (test hyperparams)
#         val_loss (float): The validation loss obtained at the end of training.
#         test_acc (float): The test accuracy obtained at the end of training.
#         test_error (float): The test error % obtained at the end of training.
#         test_abs_error (float): The test abs error % obtained at the end of training.
#         train_timer (float): Time taken to train the model.
#         mean_model_load_timer (float): Average time taken to load the model.
#         mean_model_timer (float): Average time taken to use the model for prediction.
#         try_hyperparams (ModelHyperparameters): An instance containing the hyperparameters used for this training attempt.
#         mode (Mode): The mode of the model .
#         epoch (int): The number of epochs completed at the end of training.
#         criterion_name (str): Name of the criterion (loss function) used.
#         criterion_params (dict): Parameters of the criterion used.
#     """

#     text = (
#         f"num_try = {num_try}\n"
#         f"val_loss = {val_loss}\n"
#         f"test_acc = {test_acc}\n"
#         f"test_error = {test_error}\n"
#         f"test_abs_error = {test_abs_error}\n"
#         f"execution_time_train = {train_timer}\n"
#         f"execution_time_load_saved_model = {mean_model_load_timer}\n"
#         f"execution_time_use_saved_model = {mean_model_timer}\n"
#         f"mode = {mode}\n"
#         f"batch_size = {try_hyperparams.batch_size}\n"
#         f"n_nodes = {try_hyperparams.n_nodes}\n"
#         f"activations = {try_hyperparams.activations}\n"
#         f"L1_penalty = {try_hyperparams.L1_penalty}\n"
#         f"L2_penalty = {try_hyperparams.L2_penalty}\n"
#         f"learning_rate = {try_hyperparams.learning_rate}\n"
#         f"dropout_prob = {try_hyperparams.dropout_prob}\n"
#         f"use_batch_norm = {try_hyperparams.use_batch_norm}\n"
#         f"num_epochs_used = {epoch}\n"
#         f"criterion_name = {criterion_name}\n"
#         f"criterion_params = {criterion_params}"
#     )

#     save_text_to_file(text, f"{file_path}/model_informations.txt")


# def read_info_model(file_path, infos):
#     """
#     Extract specified information from a model information file.

#     This function reads a text file containing model information in a key-value format,
#     extracts the values for the specified keys, and returns them as a list. If the value
#     can be converted to a float, it is automatically converted.

#     Args:
#     - file_path (str): The path to the file containing model information.
#     - infos (list of str): A list of keys for which values need to be extracted from the file.

#     Returns:
#         list: A list of extracted values corresponding to the specified keys in the `infos` list.
#     """
#     extracted_values = []

#     with open(file_path, "r") as file:
#         for line in file:
#             # Split the line into key and value
#             parts = line.split(" = ")
#             key, value = parts
#             value = value.strip().strip("\n")

#             # Check if the key is in the infos list
#             if key in infos:
#                 # Try to convert value to a number, if possible
#                 try:
#                     value = float(value)
#                 except ValueError:
#                     pass
#                 extracted_values.append(value)

#     return extracted_values


# def get_min_value_from_csv(csv_path, col):
#     """
#     Reads a CSV file, retrieves a specific column, sorts it in ascending order,
#     and returns the smallest value.

#     Args:
#         csv_path (str): The path to the CSV file.
#         col (str): The name of the column to retrieve and sort.

#     Returns:
#         The smallest value in the specified column.

#     Raises:
#         ValueError: If the specified column does not exist in the CSV.
#     """

#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(csv_path)

#     # Check if the specified column exists in the DataFrame
#     if col not in df.columns:
#         raise ValueError(f"The column '{col}' does not exist in the CSV file.")

#     # Sort the column in ascending order and get the smallest value
#     min_value = df[col].min()

#     return min_value
