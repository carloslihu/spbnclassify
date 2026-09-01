import concurrent.futures
import json
import re
import time
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from time import time

import pandas as pd

from ..data_handler import DataHandler
from ..module import CompositeModule
from .model import classifier_wrapper_dict
from .thresholder import thresholder_dict


class Pipeline(CompositeModule):
    """An abstract class to manage and execute a sequence of data processing and machine learning steps.

    The Pipeline class is designed to streamline the process of applying a series of transformations
    and machine learning models to a dataset. It allows for the sequential execution of various
    stages, including data preprocessing, feature extraction, model training, and evaluation.

    Attributes:
        config_dict (dict): Configuration dictionary defining the pipeline structure and parameters.
        module_id (str): Identifier for the pipeline instance.
        level (int): The hierarchical level of the pipeline in a composite structure.

    Methods:
        get_parameters(reduced_format: bool): Retrieves and flattens parameters from the configuration dictionary.
        get_num_parameters(): Calculates the total number of parameters in the pipeline.
        get_metrics(reduced_format: bool): Retrieves and flattens metrics from JSON files within the module path.
        get_train_status(): Determines the trained/untrained status of the pipeline.
        get_window_size(): Calculates the maximum window size of all modules in the pipeline.
        get_images(): Aggregates PNG image paths within the module path.
        train(data_handler: DataHandler): Abstract method to train the pipeline on a dataset.
        test(data_handler: DataHandler): Tests the pipeline on a dataset and aggregates results.
        transform_instance(data: pd.DataFrame): Transforms a single-row DataFrame using the pipeline.
        transform_batch(data: pd.DataFrame): Transforms a multi-row DataFrame using the pipeline.
        store(pipeline_path: Path): Stores the pipeline configuration and state to a specified path.
        from_json_file(config_file_name: str, config_file_path: Path, module_id: str): Initializes a pipeline from a JSON configuration file.
        recover(pipeline_path: Path, module_id: str): Recovers a pipeline from a stored configuration file.
        recover_latest(base_path: Path): Recovers the most recent pipeline from a given path.
    """

    def __init__(
        self,
        config_dict: dict,
        module_id: str = "timestamp",
        level: int = 0,
    ) -> None:
        """Initializes the Pipeline from a configuration dictionary. If the pipeline is to be initialized from a json path, the  classmethod .from_json_file should be used instead.

        Args:
            config_dict (dict): Dictionary containing the configuration file.
            module_id (str, optional): Identifier of the pipeline. When set to "timestamp", it will be set to the current timestamp. Defaults to "timestamp".
        """
        # Initialize the pipeline with the configuration dictionary
        super().__init__(config_dict=config_dict, module_id=module_id, level=level)
        self.num_parameters = 0
        self.window_size = 0
        # Print the pipeline configuration as a module tree if the Pipeline object is base-level and logging level is INFO
        if level == 0:
            self.log_note(-1, self._get_module_tree())
            print("\n")

    def get_parameters(self, reduced_format: bool = False) -> dict:
        """
        Retrieves and flattens parameters from the configuration dictionary.
        This method traverses the nested configuration dictionary and extracts parameters
        from sub-dictionaries that contain "module_type" and either "model_args" or "train_args" keys.
        The extracted parameters are flattened and returned as a single dictionary.
        Returns:
            dict: A dictionary containing the flattened parameters.
        """

        def extract_parameters(d: dict, prefix: str = "") -> dict:
            """
            Recursively traverses a nested dictionary to flatten specific sub-dictionaries
            based on the presence of certain keys.
            Args:
                d (dict): The dictionary to traverse.
                prefix (str, optional): The prefix to prepend to keys in the flattened dictionary. Defaults to "".
            Returns:
                dict: A dictionary with flattened parameters from sub-dictionaries that contain
                    "module_type" and either "model_args" or "train_args" keys.
            """

            flattened_params = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    # Check if the current dictionary is a ML model
                    if "module_type" in value and (
                        "model_args" in value or "train_args" in value
                    ):
                        if reduced_format:
                            prefix = ""
                        else:
                            prefix = key + "/"
                        for args_key in ["model_args", "train_args"]:
                            if args_key in value:
                                flattened_params.update(
                                    self.flatten_dict(value[args_key], prefix=prefix)
                                )
                    else:
                        flattened_params.update(extract_parameters(value, prefix))
            return flattened_params

        flattened_params = extract_parameters(self.config_dict)
        return flattened_params

    def get_num_parameters(self) -> int:
        """Calculates the number of parameters in the pipeline. Please do not mistake this with get_parameters (we know, poor naming).

        Args: None

        Returns:
            summed_parameters: The number of parameters in the pipeline.
        """

        def __sum_nested_dict(d: dict[str, int | dict]) -> int:
            return sum(
                __sum_nested_dict(v) if isinstance(v, dict) else v for v in d.values()
            )

        module_parameters = self._broadcast_execute(
            "_get_num_parameters", method_absent_strategy="propagate"
        )
        summed_parameters = __sum_nested_dict(module_parameters)

        return summed_parameters

    def get_metrics(self, reduced_format: bool = False) -> dict:
        """
        Retrieves and flattens metrics from JSON files within the module path.

        This method searches for 'metrics.json' files within the module path,
        flattens their nested dictionary structures, and aggregates the results into a single dictionary.
        The keys in the resulting dictionary are prefixed with the name of the directory two levels up
        from the location of each 'metrics.json' file.

        Args:
            reduced_format (bool): If True, the keys in the resulting dictionary will be reduced to the last four parts of the path.

        Returns:
            dict: A dictionary containing the flattened metrics with prefixed keys.
        """

        flattened_metrics = {}

        # We load from the metrics.json file in the output directory of the module
        file_path = self.module_path / "output/metrics.json"
        # Get the name of the directory two levels up
        module_id = file_path.parent.parent.name
        pipeline_id = file_path.parent.parent.parent.name
        prefix = f"{pipeline_id}/{module_id}"
        with file_path.open("r") as f:
            data_dict = json.load(f)

        flattened_dict = self.flatten_dict(data_dict)

        for key, value in flattened_dict.items():
            new_key = f"{prefix}/{key}"
            if reduced_format:
                new_key = "/".join(new_key.split("/")[-4:])
            flattened_metrics[new_key] = value

        return flattened_metrics

    def get_image_path_dict(self) -> dict[Path, Path]:
        """
        Searches for PNG paths within the module path and aggregates the results into a single dictionary.
        The keys in the resulting dictionary are prefixed with the name of the directory four levels up
        from the location of each image.

        Returns:
            dict: A dictionary containing the image paths with prefixed keys.
        """
        # RFE: Rethink how we store our paths, the last 5 parts of the path are not always useful
        image_path_dict = {}
        # NOTE: We only look for images in the test_full folder
        for local_image_path in self.module_path.rglob("test_full/*.png"):
            remote_image_path = Path(*local_image_path.parts[-5:])
            image_path_dict[local_image_path] = remote_image_path

        # NOTE: We look for all pdf
        for local_image_path in self.module_path.rglob("*.pdf"):
            remote_image_path = Path(*local_image_path.parts[-5:])
            image_path_dict[local_image_path] = remote_image_path

        return image_path_dict

    def get_train_status(self) -> int:
        """Calculates the trained/untrained status of the whole pipeline as the boolean multiplication of the status of all the modules.

        Args: None

        Returns:
            train_status: The window size of the pipeline.
        """

        def __all_nested_dict(d: dict[str, int | dict]) -> int:
            return all(
                __all_nested_dict(v) if isinstance(v, dict) else v for v in d.values()
            )

        module_train_status = self._broadcast_execute(
            "_get_train_status", method_absent_strategy="propagate"
        )
        train_status = __all_nested_dict(module_train_status)

        return train_status

    def get_window_size(self) -> int:
        """Calculates the window size of the whole pipeline as the maximum window size of all the modules.

        Args: None

        Returns:
            max_window: The window size of the pipeline.
        """

        def __max_nested_dict(d: dict[str, int | dict]) -> int:
            if d:
                return max(
                    __max_nested_dict(v) if isinstance(v, dict) else v
                    for v in d.values()
                )
            else:
                return 0

        module_windows = self._broadcast_execute(
            "_get_window_size", method_absent_strategy="propagate"
        )
        max_window = __max_nested_dict(module_windows)

        return max_window

    @abstractmethod
    def train(
        self, data_handler: DataHandler, should_retrain: bool = False
    ) -> pd.DataFrame | dict[str, pd.DataFrame] | None:
        """Trains the pipeline on the specified dataset.

        Args:
            data_handler (DataHandler): DataHandler object containing the dataset to train the pipeline on.
            should_retrain (bool): Whether to retrain the pipeline if it has already been trained. Defaults to False.
        Returns:
            pd.DataFrame | dict[str, pd.DataFrame] | None: A DataFrame with the validation results or a dictionary with predictions for each test part.
            If the pipeline does not have a validation step, it returns None.
        """
        pass

    def test(
        self, data_handler: DataHandler, skip_full_test: bool = False
    ) -> dict[str, pd.DataFrame]:
        """Tests the pipeline on the specified dataset.

        Args:
            data_handler (DataHandler): DataHandler object containing the dataset to test the pipeline on.
        Returns:
            dict[str, pd.DataFrame]: A dictionary containing the dataframes with the predictions for each test part.
        """
        # RFE: Aggregate all metrics in a single json file
        # TODO: resetear el estado del pipeline después de cada test (del CUSUM p.e.)
        # RFE: opción de no volver a calcular todos los score (sólo regenerar las métricas)
        # Announce the beginning of the testing process
        self.log_note(
            self.level,
            f"Testing {self.module_id} on dataset {data_handler.dataset_name}",
        )
        # Store the metric results in the result_dict
        result_dict = {}

        # Test the pipeline on each of the data handler test parts
        for test_name, test_part in getattr(data_handler, "test_data_dict", {}).items():
            # If skip_full_test is True, skip the full test
            if skip_full_test and "_full" in test_name:
                continue
            # Copy test data to avoid modifying the original data
            test_data = test_part.copy()
            # This returns the result of the test, which is a DataFrame with the predictions
            prediction_df = self._series_execute(
                "evaluate", data=test_data, test_name=test_name
            )
            result_dict[test_name] = prediction_df

        # NOTE: This is needed for test results recovery
        _ = self._join_metrics()

        return result_dict

    def transform_instance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input data using the pipeline. Admits a single-row dataframe as input (single-instance).

        Args:
            data (pd.DataFrame): Input data to transform.

        Returns:
            pd.DataFrame: Transformed data.
        """
        return self._series_execute("transform_instance", data=data)

    def transform_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input data using the pipeline. Admits a DataFrame as input (multi-instance).

        Args:
            data (pd.DataFrame): Input data to transform.

        Returns:
            pd.DataFrame: Transformed data.
        """
        return self._series_execute("transform_batch", data=data)

    def load_module(self, module_path: Path, is_full_path: bool = False) -> None:
        """Loads the pipeline and refreshes derived aggregate attributes.

        Args:
            module_path (Path): Base path where the pipeline is saved.
            is_full_path (bool, optional): Whether ``module_path`` already includes the
                pipeline directory. Defaults to False.
        """
        super().load_module(module_path, is_full_path)
        self.num_parameters = self.get_num_parameters()
        self.window_size = self.get_window_size()

    def store(self, pipeline_path: Path) -> None:
        """Stores the pipeline in a specified path.

        Args:
            pipeline_path (str): Path where the pipeline will be stored.
        """
        # Save the pipeline using the inherited method
        self.save_module(pipeline_path)

    @classmethod
    def recover_latest(cls, base_path: Path) -> "Pipeline":
        """Recovers the most recent pipeline from a given path.

        Args:
            base_path (Path): Path where the configuration files are stored.

        Raises:
            ValueError: If there are no timestamped stored pipelines in the given path.

        Returns:
            Pipeline: The most recent pipeline.
        """
        # TODO: que anuncie el recovery no como "initializing" sino como "recovering"
        file_names = list(base_path.iterdir())

        # Regular expression to find timestamps in file_names
        timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}")

        # Extract timestamp and model name from file_names
        most_recent_model = None
        most_recent_timestamp = None

        for file_path in file_names:
            match = timestamp_pattern.search(file_path.name)
            if match:
                timestamp_str = match.group()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H:%M:%S")
                if most_recent_timestamp is None or timestamp > most_recent_timestamp:
                    most_recent_timestamp = timestamp
                    most_recent_model = file_path

        if most_recent_model is None:
            raise ValueError(
                "There are no timestamped stored pipelines in the given path"
            )

        return cls.recover(most_recent_model, module_id=most_recent_model.name)

    def _train_elements(
        self,
        dataset_name: str,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        should_retrain: bool = False,
    ) -> pd.DataFrame | None:
        """Trains the pipeline elements on the specified dataset.

        Args:
            dataset_name (str): Name of the dataset to train on.
            train_data (pd.DataFrame): Training data.
            val_data (pd.DataFrame): Validation data.
        Returns:
            pd.DataFrame: The result of the validation process, which is a DataFrame with the predictions.
        """
        # Announce the beginning of the training process
        self.log_note(
            self.level,
            f"Training {self.module_id} on dataset {dataset_name}",
        )
        # Train the pipeline elements
        _ = self._broadcast_execute(
            "learn",
            dataset_name=dataset_name,
            data=train_data,
            should_retrain=should_retrain,
        )

        # RFE: don't proceed with validation if some learning process failed (e.g. OOM error on first epoch of a model)
        result = None
        if val_data is not None:
            # Announce the beginning of the validation process
            self.log_note(
                self.level,
                f"Validating {self.module_id} on dataset {dataset_name}",
            )
            # Fit the elements that need adjustment, transform the data using the rest
            result = self._series_execute(
                "fit_transform",
                alternate_method="transform_batch",
                data=val_data,
            )

        self.num_parameters = self.get_num_parameters()
        self.window_size = self.get_window_size()

        return result


# Add this function at the module level (outside any class)
def _cross_validate_worker(pipeline, cv_data_handler):
    """Worker function for cross-validation that can be pickled."""
    # NOTE: Intermediate models are not saved
    pipeline.train(cv_data_handler, should_retrain=True)
    pipeline.test(cv_data_handler)


class ClassifierPipeline(Pipeline):
    """ClassifierPipeline is a specialized implementation of the Pipeline class designed for training
    classification models. It integrates various modules for classification and thresholding,
    allowing for flexible and configurable pipelines.
    Attributes:
        modules_dict (dict): A dictionary containing the modules used in the pipeline. It combines
            classifier_wrapper_dict and thresholder_dict, which define the components of the pipeline.
    Methods:
        train(data_handler: DataHandler) -> pd.DataFrame:
            Trains the pipeline on the specified dataset. This method separates the training and
            validation data and calls the parent class's `_train_elements` method to execute the
            training process.
    """

    modules_dict = {
        **classifier_wrapper_dict,
        **thresholder_dict,
    }

    # metric_set_class = None

    def train(
        self, data_handler: DataHandler, should_retrain: bool = False
    ) -> pd.DataFrame | None:
        """Trains the pipeline on the specified dataset.

        Args:
            data_handler (DataHandler): DataHandler object containing the dataset to train the pipeline on.
        """
        # Separate training and validation data
        train_data = data_handler.train_attack_data

        # Calls the train method of the general pipeline on the split data
        result_df = super()._train_elements(
            dataset_name=data_handler.dataset_name,
            train_data=train_data,
            should_retrain=should_retrain,
        )
        return result_df

    def cross_validate(
        self,
        data_handler: DataHandler,
        n_splits: int = 5,
        n_repeats: int = 1,
        seed: int = 0,
        max_workers: int = 0,
    ) -> pd.DataFrame:
        """
        Perform cross-validation on the provided data using repeated stratified K-fold splitting,
        aggregate metrics across folds, and compute cross-validation average metrics.
        This method splits the data into training and test sets for each fold, evaluates the model,
        collects metrics, and summarizes the results into a DataFrame and a JSON file.
            data_handler (DataHandler): An instance of DataHandler containing the dataset to be used for cross-validation.
            n_splits (int, optional): Number of folds for Stratified K-Fold. Defaults to 5.
            n_repeats (int, optional): Number of times cross-validation is repeated. Defaults to 1.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            pd.DataFrame: DataFrame summarizing the mean metric values across all cross-validation folds, grouped by class name.
        Side Effects:
            - Saves the cross-validation average metrics as a JSON file in the module's output directory.
            - Updates internal metric dictionaries with cross-validation results.
        """

        def metric_dict_to_dataframe(metric_dict: dict) -> pd.DataFrame:
            """
            Converts a nested metric dictionary into a pandas DataFrame.
            The input dictionary is expected to have the following structure:
            {
                model_id: {
                    fold_name: {
                        metric_name: {
                            class_name: {
                                "metric_value_1": value1,
                                "metric_value_2": value2,
                                ...
                            }
                        }
                    }
                }
            }
            The function flattens this structure into a DataFrame where each row corresponds to a unique combination of
            fold name and class name, with metric values as columns. Duplicate rows (based on fold name and class name)
            are removed by keeping the first occurrence.
            Args:
                metric_dict (dict): Nested dictionary containing metric results.
            Returns:
                pd.DataFrame: Flattened DataFrame with metric values indexed by fold name and class name.
            """

            flat_records = []
            for model_id, folds in metric_dict.items():
                for fold_name, metrics in folds.items():
                    for metric_name, metric_list in metrics.items():
                        for class_name, metric_values in metric_list.items():
                            record = {
                                # "model_id": model_id,
                                "fold_name": fold_name,
                                # "metric_name": metric_name,
                                "class_name": class_name,
                            }
                            record.update(metric_values)
                            flat_records.append(record)

            df = pd.DataFrame(flat_records)
            # Remove duplicated rows by dropping rows NaN values
            df = df.groupby(["fold_name", "class_name"]).first()
            return df

        cv_data_handler_list = data_handler.get_cross_validation_data_handlers(
            n_splits=n_splits, n_repeats=n_repeats, seed=seed
        )

        if max_workers == 0:
            # If max_workers is 0, we run the training and testing sequentially
            for cv_data_handler in cv_data_handler_list:
                _cross_validate_worker(self, cv_data_handler)
        else:
            # Measure total multiprocessing clock time
            start_mp_time = time()
            # Use ProcessPoolExecutor with the module-level function
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(_cross_validate_worker, self, cv_data_handler)
                    for cv_data_handler in cv_data_handler_list
                ]
                # Wait for all futures to complete and handle exceptions
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.log_note(
                            self.level,
                            f"An error occurred during cross-validation: {e}",
                        )
                total_mp_time = time() - start_mp_time
                self.log_note(
                    self.level,
                    f"Total cross-validation multiprocessing time: {total_mp_time:.2f}s",
                )

        # Collect metrics from all folds to add cross_validation_avg and create a dataframe summarizing the results
        combined_metric_dict = self._join_metrics()
        metric_df = metric_dict_to_dataframe(combined_metric_dict)
        # Filter for fold_name that starts with "cross_validation_fold_"
        metric_df = metric_df.loc[
            metric_df.index.get_level_values("fold_name").str.startswith(
                "cross_validation_fold_"
            )
        ]
        # Convert the DataFrame to a dictionary format for cross-validation aggregation
        average_metric_df = metric_df.groupby(["class_name"]).mean()
        std_metric_df = metric_df.groupby(["class_name"]).std()
        cross_val_avg_dict = average_metric_df.to_dict(orient="index")
        cross_val_std_dict = std_metric_df.to_dict(orient="index")

        # Formatting: Remove the last level key if it is NaN
        cross_val_avg_dict = {
            "avg_metrics": {
                k: {ik: iv for ik, iv in v.items() if pd.notna(iv)}
                for k, v in cross_val_avg_dict.items()
            },
            "std_metrics": {
                k: {ik: iv for ik, iv in v.items() if pd.notna(iv)}
                for k, v in cross_val_std_dict.items()
            },
        }

        # Metric saving
        output_path = self.module_path / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        self._dump_object(
            cross_val_avg_dict,
            self.module_path,
            "cross_validation_avg_metrics",
            Path("output"),
            force_format="json",
        )
        # We update the combined_metric_dict with the cross-validation average
        combined_metric_dict = self._join_metrics()
        return metric_df
