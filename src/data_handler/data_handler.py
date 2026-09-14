from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.utils import Bunch
from ucimlrepo import fetch_ucirepo

from ..module.module import Module
from .constants import SKLEARN_DATASET_NAME_DICT, UCI_DATASET_NAME_DICT
from .preprocessing import TRUE_CLASS_LABEL


# RFE: que los formatter sean parte del DataHandler y luego los pipelines le pasen la especificación de columnas?
# RFE: Que DataHandler herede de Module
class DataHandler:
    """DataHandler class, to load and distribute the datasets. It is treated like a dictionary. It loads the dataset parts from the DataFrame objects and stores them in the class attributes.
    There are three main parts of the dataset: train_benign_data, train_attack_data and test_data_dict (can be associated to multiple datasets).
    The class can load all of them or exclude some of them.
    The datasets are searched for in the directory specified by the data_directory_path, and the dataset_name is used to find the correct dataset directory.

    Attributes:
        dataset_name (str): Name of the dataset
        data_part_keys (list[str]): list of available parts of the dataset
        data_fraction_parts (dict): Dictionary containing the fractions to reduce the dataset parts. The keys are the dataset parts and the values are the fractions (0 to 1)

        train_benign_data (pd.DataFrame): DataFrame containing the anomaly data
        train_attack_data (pd.DataFrame): DataFrame containing the classifier data
        test_data_dict (dict[str, pd.DataFrame]): Dictionary containing the test data. The keys are the test data part names and the values are the DataFrames

    """

    def __init__(
        self,
        dataset_name: str,
        data_part_dict: dict,
        data_fraction_parts: dict[str, float] = {},
        sanitize_columns: bool = True,
    ):
        """
        Initializes the DataHandler object with the dataset name and the data part dictionary.
        Args:
            dataset_name (str): The name of the dataset being handled.
            data_part_dict (dict): A dictionary containing the data parts, where keys are part names
                and values are the corresponding data.
            data_fraction_parts (dict[str, float], optional): A dictionary specifying the fraction
                to reduce certain data parts. Keys are part names, and values are fractions in the
                range (0, 1.0]. Defaults to an empty dictionary.
        Raises:
            ValueError: If any fraction in `data_fraction_parts` is not in the interval (0, 1.0].
        Attributes:
            dataset_name (str): The name of the dataset being handled.
            data_fraction_parts (dict[str, float]): The dictionary specifying fractions to reduce
                certain data parts.
            data_part_list (list[str]): A list of available data parts, including any individual parts from
                "test_data_dict" if present.
        """

        self.dataset_name = dataset_name
        self.data_part_keys = list(data_part_dict.keys())
        self.data_fraction_parts = data_fraction_parts

        # set placeholder for optional preprocessing steps
        self.preprocessing_steps = []
        # respect caller preference about sanitizing column names
        self.sanitize_columns = sanitize_columns

        # Check if the fraction to reduce the train_benign_data dataset is between 0 and 1
        for part, fraction in self.data_fraction_parts.items():
            if fraction <= 0 or fraction > 1:
                raise ValueError(
                    f"The fraction to reduce the {part} dataset must be in the interval (0, 1.0]"
                )
        for data_part_name, data_part in data_part_dict.items():
            if data_part_name == "test_data_dict":
                for test_key, test_df in data_part.items():
                    # NOTE: Data sanitization is done here to ensure that the column names are valid variable names
                    if self.sanitize_columns:
                        test_df.columns = Module.sanitize_variable_name_list(
                            test_df.columns
                        )

                # If there is test_data_dict, separate the test_data_dict dictionary into different data parts
                self.test_data_dict = data_part
                self.data_part_keys.remove("test_data_dict")
                self.data_part_keys += list(self.test_data_dict.keys())
            else:
                # NOTE: Data sanitization is done here to ensure that the column names are valid variable names
                if self.sanitize_columns:
                    data_part.columns = Module.sanitize_variable_name_list(
                        data_part.columns
                    )
                if data_part_name == "train_benign_data":
                    self.train_benign_data = data_part
                    # Loading train_benign_data and selecting the only entry in the retrieved dictionary
                    if (
                        data_part_name in self.data_fraction_parts
                        and self.data_fraction_parts[data_part_name] < 1
                    ):
                        Module.log_note(
                            1,
                            f"Returning the {self.data_fraction_parts[data_part_name]} fraction of the {data_part_name} dataset",
                        )
                        # Reduce the train_benign_data dataset to the specified fraction, preserving the order (for temporal models)
                        self.train_benign_data = self.train_benign_data[
                            : int(
                                len(self.train_benign_data)
                                * self.data_fraction_parts[data_part_name]
                            )
                        ]
                elif data_part_name == "train_attack_data":
                    # Loading train_attack_data and selecting the only entry in the retrieved dictionary
                    self.train_attack_data = data_part
                    if (
                        data_part_name in self.data_fraction_parts
                        and self.data_fraction_parts[data_part_name] < 1
                    ):
                        Module.log_note(
                            1,
                            f"Returning the {self.data_fraction_parts[data_part_name]} fraction of the {data_part_name} dataset",
                        )
                        # RFE: esta parte tendría que ir gestionada desde la producción del dataset para poder hacer un bucle que itere sobre "train_attack_data", "train_benign_data" y "test_data_dict"
                        # Stratified train-test split for the train_attack_data dataset
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.train_attack_data[
                                self.train_attack_data.columns.difference(
                                    [TRUE_CLASS_LABEL]
                                )
                            ],
                            self.train_attack_data[TRUE_CLASS_LABEL],
                            train_size=self.data_fraction_parts[data_part_name],
                            stratify=self.train_attack_data[TRUE_CLASS_LABEL],
                        )
                        self.train_attack_data = pd.concat([X_train, y_train], axis=1)
                else:
                    # If the data part is not recognized, raise an error
                    raise ValueError(
                        f"Data part {data_part_name} is not recognized. Allowed parts are train_benign_data, train_attack_data and test_data_dict."
                    )

    def __getitem__(self, retrieved_item: str):
        """
        Retrieve a specific part of the dataset as a pandas DataFrame or Series, or the series of preprocessing steps
        Parameters:
            retrieved_item (str): The name of the data part to retrieve. It can be one of the
                            predefined attributes ("train_benign_data", "train_attack_data", "test_data_dict")
                            or a key within the "test_data_dict" DataFrame.
        Returns:
            pd.DataFrame: The requested data part as a pandas DataFrame or Series.
        Raises:
            ValueError: If the specified data part is not found in the dataset.
        """
        # RFE: todo esto de los métodos custom de __getitem__ y __setitem__ que he metido está un poco guarro..... Deberíamos dejarlo más bonito
        if retrieved_item in [
            "train_benign_data",
            "train_attack_data",
            "test_data_dict",
        ]:
            return getattr(self, retrieved_item)
        else:
            if retrieved_item in getattr(self, "test_data_dict", {}):
                return self.test_data_dict[retrieved_item]
            elif (
                retrieved_item == "preprocessing_steps"
                or retrieved_item in self.preprocessing_steps
            ):
                return getattr(self, retrieved_item)
            else:
                raise ValueError(f"Data part {retrieved_item} not found in the dataset")

    def __setitem__(self, data_part: str, value: pd.DataFrame):
        """
        Sets the value of a specified data part within the object.
        Parameters:
            data_part (str): The name of the data part to set. It can be one of
                            ["train_benign_data", "train_attack_data", "test_data_dict"] or
                            a key within the `test_data_dict` attribute.
            value (pd.DataFrame): The DataFrame to assign to the specified data part.
        Raises:
            ValueError: If `data_part` is not in ["train_benign_data", "train_attack_data", "test_data_dict"]
                        and is not a key within the `test_data_dict` attribute.
        """
        available_parts = ["train_benign_data", "train_attack_data", "test_data_dict"]
        embedded_parts = [element + "_embedded" for element in available_parts]
        test_data_embedded = []
        if getattr(self, "test_data_dict", False):
            test_data_embedded = [
                element + "_embedded" for element in self.test_data_dict
            ]
        if data_part in available_parts or data_part in embedded_parts:
            setattr(self, data_part, value)
        else:
            if data_part in self.test_data_dict or data_part in test_data_embedded:
                self.test_data_dict[data_part] = value
            else:
                raise ValueError(f"Data part {data_part} not found in the dataset")

    @classmethod
    def from_path(
        cls,
        dataset_name: str,
        data_directory_path: Path,
        data_part_list: list[str] = [
            "train_benign_data",
            "train_attack_data",
        ],
        data_fraction_parts: dict[str, float] = {},
        sanitize_columns: bool = True,
    ) -> "DataHandler":
        """Initializes the DataHandler class, loading the dataset parts from the pickle files and storing them in the class attributes.

        Args:
            dataset_name (str): Name of the to-be-loaded dataset
            data_directory_path (str): Directory where the datasets are stored.
            data_part_list (list[str], optional): list of parts of the dataset to include. Can be "train_benign_data", or "train_attack_data". Defaults to ["train_benign_data", "train_attack_data"]
            data_fraction_parts (dict, optional): Dictionary containing the fractions to reduce the dataset parts. The keys are the dataset parts and the values are the fractions (0 to 1). Defaults to {} (no reductions)
        """

        def __load_dataset_parts(
            dataset_path: Path,
            dataset_parts: list[str],
        ) -> dict:
            """Loads a dictionary of dataset parts given a checked dataset directory.

            Args:
                dataset_path (Path): Path pointing to the directory of the dataset containing the to-be-loaded part
                dataset_parts list[str]: Parts of the dataset to load. Can be chosen to be "train_benign_data", "train_attack_data", "test", ...

            Raises:
                ValueError: Dataset part does not exist

            Returns:
                dict: Dictionary where the keys are dataset part names and the values are pd.DataFrame objects or dictionaries of pd.DataFrame objects
            """
            df_dict = {}
            # Iterate over the dataset parts and load them
            for dataset_part in dataset_parts:
                # Extract the dataset name from the dataset part (removing the pkl extension)
                dataset_part_name = dataset_part.split(".")[0]
                # Compose the path to the dataset part
                dataset_part_path = Path(f"{dataset_path}/{dataset_part}")
                # Check if the dataset part exists
                if dataset_part_path.exists():
                    # If it does, load it
                    df = pd.read_pickle(dataset_part_path)
                    # NOTE: Data sanitization is done here to ensure that the column names are valid variable names
                    if sanitize_columns:
                        df.columns = Module.sanitize_variable_name_list(df.columns)
                    # Append the loaded dataset part to the dictionary
                    df_dict[dataset_part_name] = df
                else:
                    Module.log_note(
                        1,
                        f"Dataset part {dataset_part} does not exist in path {dataset_part_path}",
                        category="WARNING",
                    )

            # Return the dictionary containing the loaded dataset parts
            return df_dict

        def __find_dataset_parts(
            dataset_path: Path, search_string: str = ""
        ) -> list[str]:
            """Finds the available parts of a dataset, given a checked dataset directory, and filters them by a search string.
            If no search string is provided, all of the dataset parts are returned.

            Args:
                dataset_path (Path): Path pointing to the to-be-loaded dataset parts
                search_string (str, optional): String to filter the dataset parts. Defaults to "" (no filtering)

            Returns:
                list[str]: list of dataset_parts including the search string
            """

            dataset_parts = []
            # Iterate over the files in the dataset directory
            for file in dataset_path.iterdir():
                if file.is_file() and (
                    search_string == "" or search_string in file.name
                ):
                    dataset_parts.append(file.name)
            # Sort the dataset parts for consistency
            dataset_parts = sorted(dataset_parts)
            return dataset_parts

        # Compose and check the dataset path from the data_directory_path and the dataset_name
        dataset_path = data_directory_path / dataset_name
        # Check if the dataset directory where the pkl files are stored exists
        if not dataset_path.exists():
            raise ValueError(
                f"Dataset {dataset_name} does not contain pickle files to load"
            )
        Module.log_note(1, f"Loading dataset parts from {data_directory_path}")
        # RFE: se podría hacer mejor lo de devolver el diccionario de partes de forma genérica y tomar "la única entrada" para anomaly/classifier _data?
        # RFE: más seguridad para no poner nombres erróneos en exclude y reduce parts?
        # RFE: hacer en un único loop todo esto? -> Entonces el reduce se haría igual para anomaly y class...
        data_part_dict = __load_dataset_parts(
            dataset_path,
            [element + ".pkl" for element in data_part_list],
        )
        # Finding all of the parts that correspond to test data
        dataset_test_parts = __find_dataset_parts(dataset_path, search_string="test")
        data_part_list += dataset_test_parts
        # RFE: qué pasa si sólo hubiera una parte de test?
        # Loading test_data_dict and storing the whole retrieved dictionary (there usually are N>1 test parts)
        data_part_dict["test_data_dict"] = __load_dataset_parts(
            dataset_path, dataset_test_parts
        )

        if len(data_part_dict["test_data_dict"]) > 1:
            # Join the existing test parts into a single dictionary entry (if there are test parts)
            data_part_dict["test_data_dict"]["test_full"] = pd.concat(
                data_part_dict["test_data_dict"].values(), ignore_index=True
            )

        # Initialize the DataHandler class with the loaded dataset parts
        return cls(
            dataset_name,
            data_part_dict,
            data_fraction_parts,
            sanitize_columns=sanitize_columns,
        )

    @classmethod
    def from_public_data(
        cls,
        dataset_name: str,
        data_fraction_parts: dict[str, float] = {},
        seed: int = 0,
        cross_validation_mode: bool = False,
        max_train_data_size: int = 0,
    ) -> "DataHandler":
        # Initialize the DataHandler class with the loaded dataset parts
        def load_public_dataset(dataset_name: str) -> Bunch | tuple:
            def load_ucirepo_dataset(dataset_name: str) -> Bunch:
                """Loads a dataset from the UCI Machine Learning Repository.

                Args:
                    dataset_name (str): Name of the dataset to load.

                Returns:
                    Bunch: A Bunch object containing the dataset features and targets.
                """

                uci_dict = fetch_ucirepo(id=UCI_DATASET_NAME_DICT[dataset_name])
                if uci_dict.data is None:
                    raise ValueError(
                        f"No data found for dataset '{dataset_name}' from UCI repository"
                    )
                dataset_dict = Bunch()
                dataset_dict.data = uci_dict.data.features  # type: ignore
                # Remove duplicated columns in the dataset
                dataset_dict.data = dataset_dict.data.loc[
                    :, ~dataset_dict.data.columns.duplicated()
                ]

                # NOTE: If targets has more than 1 column, pick the first object-typed column
                targets = uci_dict.data.targets
                if targets.shape[1] > 1:
                    object_columns = targets.select_dtypes(include=["object"]).columns
                    if len(object_columns) > 0:
                        dataset_dict.target = targets[
                            object_columns[0]
                        ]  # Take the first object column
                    else:
                        dataset_dict.target = targets.iloc[:, 0]
                else:
                    # If no object columns, take the first column as fallback
                    dataset_dict.target = targets.iloc[:, 0]
                return dataset_dict

            dataset_dict = Bunch()
            # scikit-learn datasets
            if dataset_name in SKLEARN_DATASET_NAME_DICT:
                dataset_dict = SKLEARN_DATASET_NAME_DICT[dataset_name](as_frame=True)
            else:  # UCI datasets
                dataset_dict = load_ucirepo_dataset(dataset_name)

            return dataset_dict

        dataset_dict = load_public_dataset(dataset_name)
        # Columns are sanitized and numeric columns are kept
        X = dataset_dict.data
        # Remove columns where all values are non-numeric
        numeric_cols = (
            X.apply(lambda col: pd.to_numeric(col, errors="coerce")).notna().any()
        )
        X = X.loc[:, numeric_cols]

        # Convert remaining columns to numeric, coerce errors to NaN
        X = X.apply(pd.to_numeric, errors="coerce")
        # Drop rows with any NaN values (i.e., rows with non-numeric values and NaN values)
        X = X.dropna(axis=0, how="any")
        X = X.astype("float32")
        X.columns = Module.sanitize_variable_name_list(X.columns)

        # Restrict y to the same rows as X (after dropping NaNs)
        y = dataset_dict.target.loc[X.index]
        y = y.astype(str).str.strip().astype("category")
        y.name = TRUE_CLASS_LABEL

        data_df = pd.concat([X, y], axis=1, ignore_index=False).reset_index(drop=True)

        # We remove the classes that have less than a certain number of samples
        # We calculate the minimum class sample size based on the training data size for 5-fold cross-validation two times
        min_class_sample_size = np.ceil(data_df.shape[1] / 0.8**2)
        label_counts = data_df[TRUE_CLASS_LABEL].value_counts()

        valid_labels = label_counts[label_counts >= min_class_sample_size].index
        removed_labels = label_counts[label_counts < min_class_sample_size].index
        if len(removed_labels) > 0:
            print(
                f"{dataset_name}: Removed {len(removed_labels)} label(s): {list(removed_labels)}"
            )
        data_df = data_df[data_df[TRUE_CLASS_LABEL].isin(valid_labels)]
        data_df[TRUE_CLASS_LABEL] = data_df[
            TRUE_CLASS_LABEL
        ].cat.remove_unused_categories()

        X_train, X_test, y_train, y_test = train_test_split(
            data_df[data_df.columns.difference([TRUE_CLASS_LABEL])],
            data_df[TRUE_CLASS_LABEL],
            test_size=0.2,
            stratify=data_df[TRUE_CLASS_LABEL],
            random_state=seed,
        )
        if cross_validation_mode:
            train_attack_data = data_df
        else:
            train_attack_data = pd.concat(
                [X_train, y_train], axis=1, ignore_index=False
            )
        # If the max_train_data_size is set, we reduce the train_attack_data to that size
        if max_train_data_size > 0 and train_attack_data.shape[0] > max_train_data_size:
            train_attack_data = train_attack_data.sample(
                n=max_train_data_size, random_state=seed, ignore_index=True
            )

        test_data = pd.concat([X_test, y_test], axis=1, ignore_index=False)

        data_part_dict = {
            "train_attack_data": train_attack_data,
            "test_data_dict": {"test_full": test_data},
        }

        return cls(dataset_name, data_part_dict, data_fraction_parts)

    def get_cross_validation_data_handlers(
        self, n_splits: int = 5, n_repeats: int = 1, seed: int = 0
    ) -> list["DataHandler"]:
        """Generates cross-validation data handlers for the dataset.

        Args:
            n_splits (int): Number of splits for cross-validation.
            n_repeats (int): Number of times to repeat the cross-validation.
            seed (int): Random seed for reproducibility.

        Returns:
            list[DataHandler]: list of DataHandler objects for each cross-validation fold.
        """

        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        X = self.train_attack_data[
            self.train_attack_data.columns.difference([TRUE_CLASS_LABEL])
        ]
        y = self.train_attack_data[TRUE_CLASS_LABEL]

        cv_data_handler_list = []
        for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
            train_data = pd.concat([X.iloc[train_index], y.iloc[train_index]], axis=1)
            test_data = pd.concat([X.iloc[test_index], y.iloc[test_index]], axis=1)
            cv_data_handler = DataHandler(
                dataset_name=f"{self.dataset_name}_cross_validation_fold_{i}",
                data_part_dict={
                    "train_attack_data": train_data,
                    "test_data_dict": {f"cross_validation_fold_{i}": test_data},
                },
            )
            cv_data_handler_list.append(cv_data_handler)

        return cv_data_handler_list
