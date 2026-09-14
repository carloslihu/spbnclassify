import json
import logging
import pickle
import re
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def make_note(
    level: int,
    note: str,
) -> str:
    """Makes a note. The note string is returned with a specific format depending on the level of the note

    Args:
        level (int): Hierarchical level of the note. Negative levels are used for notes that are not part of the hierarchical structure (no formatting).
        note (str): Note to be printed.
        wait_continuation (bool, optional): Flag to indicate if the note should wait for a continuation in the same line (the following print). Defaults to False.
    """
    if level < 0:
        string = "\n"
        string += note
        return string
    elif level == 0:
        # If the level is 0, print the note in an enclosing box
        string = "\n"
        string += " " + (6 + len(note)) * "-" + " " + "\n"
        string += "¦" + 3 * " " + note + 3 * " " + "¦" + "\n"
        string += " " + (6 + len(note)) * "-"
        return string
    else:
        # If the level is greater than 0, print the note with an arrow and a level-dependent indentation
        string = "--" * level + "> " + note
        return string


# TODO: cambiar todas las veces que se comprueba si una clave pertenece a un diccionario para recuperarla por métodos get


class Module:
    """Module class: literally the building block of the architecture.
    The Module class contains all of the basic methods and attributes that any module of the library should have.
    It is initialized by a configuration dictionary, a module ID and a level in the pipeline hierarchy. If the module ID is set to "timestamp",
    the module is assigned a timestamped ID based on the time of creation.

    Attributes:
        config_dict (dict): Dictionary with the configuration of the module.
        module_path (str): Path where the module object is saved.
        module_id (str): ID of the module.
        level (int): Hierarchical level of the module in the pipeline.

    Private Methods:
        _dump_object: Dumps an object in a specified directory.
        _load_object: Loads an object from a specified directory.

    Public Methods:
        save_module: Creates a directory for saving a module and dumps the Module config dictionary in a json file.
        load_module: Loads the module from the base path where the it had been saved.
        make_note (Static): Returns a fabricated string with a formatted note depending on the level of the note.
        log_note (Static): Logs a note with a specific format depending on the level of the note.
        camel_to_snake (Static): Converts strings from CamelCase to snake_case.

    """

    metric_set_class = None
    true_label = ""
    prediction_label = ""
    score_columns = []

    def __init__(
        self,
        config_dict: dict,
        module_id: str,
        level: int,
    ) -> None:
        """Initializes the Module class.

        Args:
            config_dict (dict): Dictionary with the configuration of the module. The structure of the dictionary depends on the particular module.
            module_id (str): ID of the module.
            level (int): Hierarchical level of the module in the pipeline.
        """

        def __set_module_id(module_id: str = "timestamp") -> None:
            """Sets the module ID. If the module ID is not provided, it is set to "timestamp". When the module_id
            is set to "timestamp", the module is assigned a timestamped ID based on the time of creation.

            Args:
                module_id (str, optional): ID of the module. Defaults to "timestamp".
            """
            if module_id == "timestamp":
                # If the module ID is set to "timestamp", assign a timestamped ID to the module
                snake_name = self.camel_to_snake(self.__class__.__name__)
                timestr = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                module_id = f"{snake_name}_{timestr}"
            # Store the module ID as an attribute
            self.module_id = module_id

        # Store the configuration dictionary as an attribute
        self.config_dict = config_dict

        self.module_path = Path()
        # set the module ID via the inner method (also stores the module ID as an attribute)
        __set_module_id(module_id)
        # Store the hierarchical level of the module in the pipeline as an attribute
        self.level = level
        # Announce the creation of the module
        self.log_note(
            self.level,
            f"Initializing {self.__class__.__name__} with ID {self.module_id}",
        )

    @classmethod
    def from_json_file(
        cls,
        config_file_name: str,
        config_file_path: Path = Path("/app/dev/rutile-ai/data/configs"),
        module_id: str = "timestamp",
        level=0,
    ):
        """Initializes the Module object by loading the configuration file from the specified path. It uses the general initialization method __init__
        after loading the configuration file.

        Args:
            config_file_name (str): Name of the configuration file to load (must be a json object).
            config_file_path (str, optional): Path where the configuration file is stored. Defaults to "/app/dev/rutile-ai/data/configs".
            module_id (str, optional): Identifier of the pipeline. When set to "timestamp", it will be set to the current timestamp. Defaults to "timestamp".
        """
        # Create an empty module object to use the _load_object method
        module = cls.__new__(cls)
        # Load the configuration file from the specified path (in json format)
        config_file = module._load_object(config_file_name, "json", config_file_path)
        # Initialize the module from the loaded configuration file
        module.__init__(
            config_file,
            module_id=module_id,
            level=level,
        )
        # If the config file was loaded (init was launched from class method .from_json_file()), announce it
        module.log_note(
            level + 1,
            f"Config file {config_file_name} loaded from {config_file_path}",
        )
        return module

    @classmethod
    def recover(cls, module_path: Path, module_id: str, is_full_path=True):
        """Recovers a module from a stored configuration file. NOTE: do not confuse with the load_module method. The latter is used to load the elements and parameters of the module, and recover is used to actually initialize the model from its stored configuration. Think of it as the load_module method being part of the recover method.

        Args:
            module_path (str): Path where the configuration file is stored.
            # TODO: que no haya que pasar el module_id como parte del path y como un argumento separado...
            module_id (str): Identifier of the module.

        Returns:
            Module: The recovered module.
        """
        config_file = f"{cls.camel_to_snake(cls.__name__)}_config"

        if not is_full_path:
            module_path = module_path / module_id

        recovered_module = cls.from_json_file(
            config_file, module_path, module_id=module_id
        )
        recovered_module.load_module(module_path, is_full_path=True)
        return recovered_module

    @staticmethod
    def camel_to_snake(name: str) -> str:
        """Converts strings from CamelCase to snake_case
        SEE: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case

            Args:
                name (str): string in CamelCase

            Returns:
                str: string in snake_case
        """
        # Substitutes all white-spaces with underscores
        name = re.sub(r"\s+", r"_", name)
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub("__([A-Z])", r"_\1", name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
        # Returns the string in lowercase
        return name.lower()

    @staticmethod
    def sanitize_variable_name(name: str) -> str:
        """
        Sanitizes a name to conform to the pattern `[^_a-z0-9]+`.

        Args:
            name (str): The name to sanitize.

        Returns:
            str: The sanitized name, converted to lowercase and with non-alphanumeric characters replaced by underscores.
        """
        return re.sub(r"[^_a-z0-9]+", "_", name.lower())

    @staticmethod
    def sanitize_variable_name_list(name_list: list) -> list:
        """
        Sanitizes a list of names to conform to the pattern `[^_a-z0-9]+`.

        Args:
            name_list (list): The list of names to sanitize.

        Returns:
            list: The sanitized names, converted to lowercase and with non-alphanumeric characters replaced by underscores.
        """
        return [Module.sanitize_variable_name(name) for name in name_list]

    @staticmethod
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: Module.convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Module.convert_to_json_serializable(i) for i in obj]
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif isinstance(obj, pd.DataFrame):
            # return obj.to_dict() # Does not handle multi-index well
            df_copy = obj.reset_index()
            if isinstance(df_copy.columns, pd.MultiIndex):
                df_copy.columns = [
                    "_".join(map(str, col)).strip("_") for col in df_copy.columns
                ]
            return df_copy.to_dict("records")
        else:
            return obj

    @staticmethod
    def flatten_dict(data_dict: dict, prefix: str = "") -> dict:
        """Recursively flattens a nested dictionary.

        Args:
            data_dict (dict): The dictionary to flatten.
            prefix (str): The prefix to prepend to the keys.

        Returns:
            dict: The flattened dictionary.
        """
        flattened_dict = {}
        for key, value in data_dict.items():
            new_key = f"{prefix}{key}"
            if isinstance(value, dict):
                flattened_dict.update(Module.flatten_dict(value, new_key + "/"))
            else:
                flattened_dict[new_key] = value
        return flattened_dict

    @staticmethod
    def update_nested_dict(nested_dict: dict, leveled_keys: list, value: str) -> None:
        """
        Updates a nested dictionary with a given value at a specified location defined by a list of keys.
        Args:
            nested_dict (dict): The dictionary to be updated. This can be a deeply nested dictionary.
            leveled_keys (list): A list of keys representing the path to the location in the dictionary
                                where the value should be set. Each key corresponds to a level in the
                                nested dictionary.
            value (str): The value to set at the specified location in the nested dictionary.

        Example:
            nested_dict = {'a': {'b': {'c': 1}}}
            leveled_keys = ['a', 'b', 'd']
            value = 2
            result = update_nested_dict(nested_dict, leveled_keys, value)
            # result is {'a': {'b': {'c': 1, 'd': 2}}}
        """

        for key in leveled_keys[:-1]:
            nested_dict = nested_dict.setdefault(key, {})
        nested_dict[leveled_keys[-1]] = value

    @staticmethod
    def log_note(
        level: int,
        note: str,
        category: str = "INFO",
    ) -> None:
        """Log a note. The note string is logged with a specific format depending on the level of the note. The log category is also specified.

        Args:
            level (int): Hierarchical level of the note.
            note (str): Note to be printed.
            category (str): Category of the log (INFO, WARNING or ERROR).
        """
        log_level = logging._nameToLevel[category]
        formatted_note = make_note(level, note)
        logging.log(log_level, formatted_note)

    @abstractmethod
    def transform_instance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms a single instance of data. This method is meant to be overwritten by the subclasses.

        Args:
            data (pd.DataFrame): Data to be transformed.

        Returns:
            pd.DataFrame: Transformed data.
        """
        pass

    def transform_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms a batch of data. This method applies the transform_instance method to each instance of the data.
        Args:
            data (pd.DataFrame): Data to be transformed.
        Returns:
            pd.DataFrame: Transformed data.
        """
        self.log_note(
            self.level,
            f"Transforming batch with {self.module_id}",
        )
        return self.transform_instance(data)

    @abstractmethod
    def evaluate(
        self,
        data: pd.DataFrame,
        test_name: str,
        batch_size: int = 0,
    ) -> pd.DataFrame:
        pass

    def save_module(
        self,
        module_path: Path = Path(),
        is_full_path: bool = False,
    ) -> None:
        """Creates a directory for saving a module and dumps the Module config dictionary in a json file.

        Args:
            module_path (str, optional): Path where the model is to be saved. Defaults to an empty string (root path).
            is_full_path (bool, optional): Flag to indicate if the full path is provided. Defaults to False.
        """
        # If the full path is not provided, append the module ID to the module path
        if not is_full_path:
            # Use the model ID as name for the directory
            module_path = module_path / self.module_id
        module_path.mkdir(parents=True, exist_ok=True)
        # Save the configuration dictionary in a json file
        self._dump_object(
            self.config_dict,
            module_path,
            f"{self.camel_to_snake(self.__class__.__name__)}_config",
        )
        # Print an announcement of the saving
        self.log_note(self.level, f"Saving {module_path}")
        # Save the model's directory path as a model property
        self.module_path = module_path

    def load_module(
        self,
        module_path: Path,
        is_full_path: bool = False,
    ) -> None:
        """Loads the module element from the base path where the submodules are saved.
        It also stores the module path as an attribute.

        Args:
            module_path (str): Base path where the composite module is saved.
            is_full_path (bool, optional): Flag to indicate if the full path is provided. Defaults to False.
            This flag is used to indicate if the module ID should be appended to the module path, and it is needed for
            every submodule in a modular structure stemming from a top-level composite module. In practice it is used
            for every module and composite module that is not the top-level pipeline.
        Returns:
            Module: The loaded module.
        """
        # If the full path is not provided, append the module ID to the module path
        if not is_full_path:
            module_path = module_path / self.module_id
        # Store the module path as an attribute
        self.module_path = module_path

    def generate_report(
        self,
        report_dict: dict,
    ) -> None:
        """Generates a report from a dictionary in a human readable format.

        Args:
            report_dict (dict): Dictionary with the report information.
        """

        def __print_item(level, item, report_file):
            if isinstance(item, dict):
                for key in item.keys():
                    report_file.write("\n")
                    if isinstance(item[key], dict):
                        report_file.write(self.__make_note(level, key).strip())
                        __print_item(level + 1, item[key], report_file)
                    else:
                        report_file.write(
                            self.__make_note(level, f"{key}: {item[key]}")
                        )
            else:
                report_file.write("\n")
                report_file.write(self.__make_note(level, str(item)))

        with open(f"{self.module_path}/report.txt", "w") as report_file:
            level = 0
            for key in report_dict.keys():
                report_file.write(self.__make_note(0, key).strip())
                __print_item(level + 1, report_dict[key], report_file)
                report_file.write("\n")

    def _calculate_metrics(
        self,
        data: pd.DataFrame,
        test_name: str,
        classes: list = [
            False,
            True,
        ],
    ) -> pd.DataFrame:
        """
        Calculates and stores evaluation metrics for the given test data.
        This method initializes the metric set using the provided data and configuration,
        computes the relevant metrics, and saves the results as a JSON file. If no metric
        set class is defined, the method logs a debug note and skips metric calculation.
        Args:
            data (pd.DataFrame): The input DataFrame containing test results and predictions.
            test_name (str): The name of the test, used for output file naming and directory structure.
            classes (list, optional): list of class labels for which to calculate metrics.
                Defaults to [False, True] for anomaly detection tasks.
        Returns:
            dict[str, Metric]: A dictionary containing the calculated metrics, where keys are metric names
            and values are Metric objects containing the results.
        """
        # Make metrics for metric_set if it has at least an element (else, skip but don't throw error since it's considered intentionally left blank)
        if self.metric_set_class is None:
            self.log_note(
                self.level,
                f"Metric set not defined for {self.module_id}, skipping metrics",
                category="DEBUG",
            )
            return pd.DataFrame()
        else:
            module_path = self.module_path / f"output/{test_name}/"
            # ???: Review with Joge
            # Check explicitly if all classes are boolean.
            # if set(classes).issubset({False, True}):
            if all(isinstance(c, (bool, np.bool_)) for c in classes):
                score_columns = ["anomaly_score"]
            else:
                score_columns = [f"{c}_score" for c in classes]
            self.metric_set = self.metric_set_class(
                data=data,
                store_path=module_path,
                true_label=self.true_label,
                prediction_label=self.prediction_label,
                score_columns=score_columns,
                level=self.level,
                metric_kwargs=self.config_dict.get("metric_kwargs", {}),
            )
            self.metric_set.calculate()
            self._dump_object(
                self.metric_set.report_dict,
                self.module_path,
                f"{test_name}_metrics",
                Path(f"output/{test_name}/"),
                force_format="json",
            )
            return self.metric_set.report_df

    def _join_metrics(self) -> dict[str, Any]:
        """
        Aggregates and returns metrics from JSON files within the module's output directory.
        If a combined metrics file already exists and `overwrite_metrics` is False, loads and returns the existing metrics.
        Otherwise, recursively searches for all files ending with '_metrics.json' in the output directory and its subdirectories, loads their contents, and combines them into a single dictionary. The combined metrics are then saved to a file.
        Args:
            overwrite_metrics (bool): If True, forces regeneration and overwriting of the combined metrics file.
                                    If False, uses the existing combined metrics file if present.
        Returns:
            dict[str, Any]: A dictionary containing the aggregated metrics, keyed by the base name of each metrics file.
        """

        output_path = self.module_path / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        # output_metrics_file_path = output_path / "metrics.json"

        # if output_metrics_file_path.exists() and not overwrite_metrics:
        #     combined_metric_dict = self._load_object("metrics", "json", output_path)
        # else:
        # We iterate through the output directory and subdirectories loading the metrics
        combined_metric_dict = {}
        for metrics_file in output_path.rglob("*_metrics.json"):
            # Get the relative path from output_path to the metrics file's parent directory
            metrics_file_relative_path = str(metrics_file.relative_to(output_path))
            key = metrics_file.name.replace("_metrics.json", "")
            combined_metric_dict[key] = self._load_object(
                metrics_file_relative_path,
                "json",
                output_path,
            )
        self._dump_object(
            combined_metric_dict,
            self.module_path,
            "metrics",
            Path("output"),
            force_format="json",
        )
        return combined_metric_dict

    def _dump_object(
        self,
        obj: Any,
        directory_path: Path,
        name: str,
        subdirectory_path: Path = Path(),
        force_format: str = "",
    ) -> None:
        """Dumps an object in a specified directory. It also allows for the creation of a subdirectory route for dumping the object.
        If the object is json-serializable, it is dumped in a json file. Otherwise, it is dumped in a pickle file.

        Args:
            object (Any): Object to be dumped.
            directory_path (str): Path where the object is to be saved.
            name (str): Name of the file where the object is to be saved.
            subdirectory_path (str, optional): Subdirectory route for saving the object. Defaults to an empty string (no subdirectory).
            force_format (str, optional): Flag to force the object to be json-serialized. Defaults to an empty string (no forced serialization). Can be "json" or "pkl".
        """

        if force_format not in ["", "json", "pkl"]:
            raise ValueError(
                f"Force format {force_format} not supported. Supported formats are 'json' and 'pkl'."
            )

        def __create_subdirectory(
            directory_path: Path, subdirectory_path: Path
        ) -> Path:
            """Creates a subdirectory in the specified directory.

            Args:
                directory_path (str): Path where the subdirectory is to be created.
                subdirectory_name (str): Composed path of the subdirectory to be created

            Returns:
                str: Composed path of the created subdirectory
            """
            # Compose the full path of the subdirectory
            subdirectory_path = directory_path / subdirectory_path
            # Create the subdirectory if it does not exist
            subdirectory_path.mkdir(parents=True, exist_ok=True)

            # Return the full path of the subdirectory
            return subdirectory_path

        def __try_json_serialization(obj: Any) -> str:
            """Tries to serialize an object in json format. If successful, it returns the serialized object. Otherwise, it returns an empty string.

            Args:
                obj (Any): Object to be serialized.

            Returns:
                str: Serialized object in json format or an empty string if the serialization fails.
            """

            try:
                # Convert the object to a json serializable format
                serializable_object = self.convert_to_json_serializable(obj)
                # Convert the serializable object to a json string
                # RFE: Check raised exceptions regarding pickle serialization of scalers
                serialized = json.dumps(serializable_object, indent=4)
                # Return the serialized object
                return serialized
            except Exception:
                # Return an empty string if the serialization fails, not json-serializable
                return ""

        if subdirectory_path:
            # Create the subdirectory route if it is provided
            directory_path = __create_subdirectory(directory_path, subdirectory_path)

        if force_format == "pkl":
            # Try to serialize the object in json format
            json_object = ""
        else:
            json_object = __try_json_serialization(obj)

        if json_object:
            # If the object is json-serializable, dump it in a json file
            with open(f"{directory_path}/{name}.json", "w") as dump_file:
                dump_file.write(json_object)
        else:
            # If the object is not json-serializable but forced to be json-serialized, raise an error
            if force_format == "json":
                raise ValueError(
                    f"Object {obj} not json-serializable. Forced json serialization failed."
                )
            # If the object is not json-serializable, dump it in a pickle file
            with open(f"{directory_path}/{name}.pkl", "wb") as dump_file:
                pickle.dump(obj, dump_file)

    def _load_object(
        self, object_name: str, extension: str, directory_path: Path = Path()
    ) -> Any:
        """Loads an object from a specified directory. The object can be loaded from a json file or a pickle file (to be specified).

        Args:
            name (str): Name of the file where the object is saved.
            extension (str): Extension of the file where the object is saved. Must be either "json" or "pkl".
            directory_path (str): Path where the object is saved.

        Raises:
            ValueError: If the extension is not supported.

        Returns:
            Any: Loaded object.
        """
        # RFE: Allow to directly load from a file path (not only from a directory)
        if extension == "json":
            # If extension is json, use json.load and standard reading permissions to load the object
            loader = json.load
            permissions = "r"
        elif extension == "pkl":
            # If extension is pkl, use pickle.load and binary reading permissions to load the object
            loader = pickle.load
            permissions = "rb"
        else:
            # If the extension is not supported, raise an error
            raise ValueError(f"Extension {extension} not supported.")
        # Remove unnecessary spacing
        object_name = object_name.strip()
        # Add the extension to object_name if not present
        if not object_name.endswith(f".{extension}"):
            object_name = f"{object_name}.{extension}"
        # Open the file with the specified permissions
        with open(directory_path / object_name, permissions) as load_file:
            # Load the object from the file
            return loader(load_file)

    @staticmethod
    def __make_note(
        level: int,
        note: str,
    ) -> str:
        """
        NOTE: this function is taken from utils.logging.log_note and set as a static method in Module for child classes to use it without importing it from utils
        Makes a note. The note string is returned with a specific format depending on the level of the note

        Args:
            level (int): Hierarchical level of the note. Negative levels are used for notes that are not part of the hierarchical structure (no formatting).
            note (str): Note to be printed.
            wait_continuation (bool, optional): Flag to indicate if the note should wait for a continuation in the same line (the following print). Defaults to False.
        """
        return make_note(level, note)
