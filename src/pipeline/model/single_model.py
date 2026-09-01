import time
from abc import abstractmethod
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler
from torch.cuda import OutOfMemoryError

from ...data_handler import TRUE_ANOMALY_LABEL, TRUE_CLASS_LABEL
from ...utils.distance import parametric_node_type_ratio
from ..evaluator.metric_set import ScoreMetricSet
from .model import Model


# RFE: Allow saving dynamic score_columns, classes, and other attributes
class SingleModel(Model):
    """
    SingleModel is an abstract base class for machine learning models within a modular pipeline architecture.
    It provides a standardized interface and common functionality for single-model modules, including model
    initialization, data scaling, training, evaluation, and information management.
    Attributes:
        metric_set_class: Class used for metric calculation.
        engine_model_class: The underlying engine model class to be instantiated.
        model_info (dict): Information about the model configuration.
        train_info (dict): Information about the training process.
        data_info (dict): Information about the data used for training and inference.
        scaler: Scaler object for data normalization or transformation.
        window_size (int): Size of the temporal window for models that require it.
        queue (list): Queue for temporal data processing.
    Methods:
        __init__(config_dict, module_id, level): Initializes the SingleModel with configuration, module ID, and hierarchy level.
        _get_window_size(): Returns the window size used by the model.
        _get_train_status(): Returns the training status of the model.
        get_info(): Returns a dictionary with model, training, and data information.
        empty_queue(): Empties the internal data queue.
        score_instance(data): Abstract method to score a single data instance.
        score_batch(data, batch_size): Abstract method to score a batch of data.
        transform_instance(data): Transforms a single data instance by adding scores and returning a scoring report dict (for anomaly models).
        transform_batch(data, batch_size): Transforms a batch of data, handling memory errors and adding scores.
        evaluate(data, test_name, batch_size): Evaluates the model on a dataset and generates metrics.
        dump_info_report_assets(): Dumps model, training, and data information in a human-readable format.
        save_module(module_path, is_full_path): Saves the model and its data information to disk.
        load_module(module_path, is_full_path): Loads the model and its data information from disk.
        _make_scaler(scaling): Creates a scaler object based on the specified scaling method.
        _scale_data(data, should_fit_scaler): Scales the input data using the scaler.
        _before_engine_learn(dataset_name, data): Prepares the model and data before training.
        _engine_data_preparation(data): Abstract method for engine-specific data preparation.
        _engine_learn(data): Abstract method for training the engine model.
        _after_engine_learn(): Finalizes the training process and updates training information.
        _on_learn(dataset_name, data): Orchestrates the full learning process (cannot be overridden).
    Note:
        This class is intended to be subclassed with concrete implementations for the abstract methods.
    """

    metric_set_class = ScoreMetricSet
    engine_model_class = None

    def __init__(self, config_dict: dict, module_id: str, level: int) -> None:
        """Initializes the SingleModel class.

        Args:
            config_dict (dict): Dictionary with the configuration of the ML Model.
            module_id (str): ID of the module.
            level (int): Hierarchical level of the module in the pipeline.

        Raises:
            ValueError: Model arguments not found in the configuration dictionary
        """
        # Initialize the SingleModel as a Module with the configuration dictionary, the module id and the level
        super().__init__(config_dict=config_dict, module_id=module_id, level=level)

        # Initialization of the model, training and data information (single-model specific)
        if "model_args" not in config_dict:
            raise ValueError(
                "Model arguments not found in the configuration dictionary"
            )
        elif self.engine_model_class is None:
            raise ValueError("engine_model_class is not set.")

        # Info initialization
        self.model_info = config_dict["model_args"].copy()
        # NOTE: train_info changes are NOT persisted
        self.train_info = config_dict.get("train_args", {})
        # NOTE: Data Info contains information PERSISTED
        # RFE: This should be moved up to Model class
        self.data_info = {}

        # Scaler initialization
        self.scaler = None

        # Window and queue initialization (generalization of all models to temporal models)
        self.window_size = self.model_info.get("window_size", 1)
        self.queue = []
        # Wrapper-level class labels used when the backend model cannot expose
        # original labels (e.g. XGBClassifier trained with encoded targets).
        self._model_classes_override = []

        # Filter model_info to only include arguments that are in the engine_model_class __init__
        model_info_filtered = {
            k: v
            for k, v in self.model_info.items()
            if k in self.engine_model_class.__init__.__code__.co_varnames
        }
        # Initialization of the ML model by using child class variable engine_class
        self.engine_model = self.engine_model_class(**model_info_filtered)

    def _get_window_size(self):
        return self.window_size

    def _get_train_status(self):
        return self.train_info["trained"]

    def get_info(self) -> dict:
        """Return a dictionary with the model, training and data information.

        Returns:
            dict: Dictionary with the model, training and data information.
        """
        return {
            "model_info": self.model_info,
            "train_info": self.train_info,
            "data_info": self.data_info,
        }

    def _get_model_classes(self, default: list = [False, True]) -> list:
        """Return class labels exposed by the wrapper/model.

        Priority:
            1. ``self._model_classes_override`` when populated.
            2. ``self.engine_model.classes_`` when available.
            3. ``default`` fallback.

        Args:
            default (list, optional): Fallback labels.

        Returns:
            list: Effective class labels for scoring/metrics.
        """
        if len(self._model_classes_override) > 0:
            return self._model_classes_override
        return getattr(self.engine_model, "classes_", default)

    def empty_queue(self) -> None:
        """Empty the queue attribute."""
        self.queue = []

    @abstractmethod
    def score_instance(self, data: pd.DataFrame) -> tuple[pd.Series, dict]:
        pass

    @abstractmethod
    def score_batch(self, data: pd.DataFrame, batch_size: int = 0) -> pd.DataFrame:
        """Scores a batch of data using the model.

        Args:
            data (pd.DataFrame): The data to be scored.
            batch_size (int, optional): The batch size for scoring. Defaults to 0.
        Returns:
            pd.DataFrame: The scores for the data.
        """
        # TODO FEB26: set safety measure against less data than window! (could make data_loader return smaller than 0 lengths)
        data = self._scale_data(data)
        return data

    def transform_instance(self, data: pd.DataFrame) -> pd.DataFrame:
        score, scoring_report = self.score_instance(
            data[list(self.data_info["data_columns"])]
        )
        # Add the score to the data as a column
        classes = self._get_model_classes()
        if all(isinstance(c, bool) for c in classes):
            score_columns = ["anomaly_score"]
        else:
            score_columns = [f"{c}_score" for c in classes]
        score_df = pd.DataFrame([score], columns=score_columns, index=data.index)
        data = pd.concat([data, score_df], axis=1)

        # Add the importances and deltas to the data as columns
        if score_columns == ["anomaly_score"]:
            # Only for anomaly scorers
            data[[element + "_imp" for element in self.data_info["data_columns"]]] = (
                np.ravel(scoring_report["importances"])
            )
            data[[element + "_delta" for element in self.data_info["data_columns"]]] = (
                np.ravel(scoring_report["deltas"])
            )
        # Append the window_size as a metadata to the data DataFrame
        data.attrs["window_size"] = self.window_size
        # Return the data with the score
        return data

    # RFE-C: Make it inherit methods from Module
    def transform_batch(
        self,
        data: pd.DataFrame,
        batch_size: int = 0,  # RFE: batch_size unused in this class, used by parents -> Redesign class?
    ) -> pd.DataFrame:
        """Transforms the data and returns it with an additional column containing the score series.

        Args:
            data (pd.DataFrame): Data to evaluate.
            batch_size (int, optional): Batch size for the evaluation. Defaults to 0.

        Returns:
            pd.DataFrame: Dataframe with the score as a column.
        """
        # Updating data information with recording information

        # Try to score the data with the given batch size
        attempted_batch_size = batch_size  # Track what batch size we're actually trying
        while True:
            try:
                # Score the data and obtain a data frame with the scores
                score_frame = self.score_batch(
                    data[list(self.data_info["data_columns"])],
                    batch_size=attempted_batch_size,
                )
                break
            except Exception as e:
                # If an OutOfMemoryError is raised, halve the batch size and try again
                if type(e) == OutOfMemoryError:
                    # If batch_size was 0 (auto), start with a reasonable size and halve it
                    if attempted_batch_size == 0:
                        attempted_batch_size = 512  # Start with a moderate size
                    elif attempted_batch_size == 1:
                        # If the batch size is already 1, raise the error
                        self.log_note(
                            self.level,
                            "Out of memory error persisting even after batch size has been set to 1.",
                            "ERROR",
                        )
                        raise e
                    else:
                        # Otherwise, halve the batch size and try again
                        attempted_batch_size = attempted_batch_size // 2

                    self.log_note(
                        self.level,
                        f"Out of memory error, trying batch_size={attempted_batch_size} for testing.",
                        "WARNING",
                    )
                else:
                    # If the error is not an OutOfMemoryError, raise it
                    raise e

        # set the index of the score frame to match the data index and concatenate it with the data
        score_frame.index = data.index
        # Concatenate the data with the score frame
        data = pd.concat([data, score_frame], axis=1)
        # Append the window_size as a metadata to the data DataFrame
        data.attrs["window_size"] = self.window_size
        # Return the data with the score
        return data

    def evaluate(
        self,
        data: pd.DataFrame,
        test_name: str,
        batch_size: int = 0,
    ) -> pd.DataFrame:
        """
        Evaluates the model on the provided data, computes metrics, and dumps the score results.
        Args:
            data (pd.DataFrame): The input data to evaluate.
            test_name (str): The name of the test or evaluation run.
            batch_size (int, optional): The batch size to use for data transformation. Defaults to 0.
        Returns:
            pd.DataFrame: The transformed data with evaluation scores.
        """
        self.log_note(
            self.level,
            f"Scoring with {self.module_id}",
        )
        # TODO: This is only saved after training, not testing
        self.data_info["testing_time"] = time.time()
        data = self.transform_batch(data, batch_size)
        self.data_info["testing_time"] = round(
            time.time() - self.data_info["testing_time"], 3
        )
        # RFE: This is too specific to the model, so it should be moved elsewhere
        # Check if the engine_model has the slogl method and log or use it if needed
        if hasattr(self.engine_model, "slogl"):
            # NOTE: This scales and types the data to float32 and category which is needed for the slogl method
            slogl_data = self._engine_data_preparation(data)
            classes = self._get_model_classes()
            slogl_data[TRUE_CLASS_LABEL] = pd.Categorical(
                slogl_data[TRUE_CLASS_LABEL], categories=classes
            )
            slogl = self.engine_model.slogl(slogl_data)
            prob_metrics_dict = {
                "probability_metrics": {
                    "macro avg": {
                        "log_likelihood": slogl,
                        "parametric_node_type_ratio": parametric_node_type_ratio(
                            self.engine_model
                        ),
                    }
                }
            }
            self._dump_object(
                prob_metrics_dict,
                self.module_path,
                f"{test_name}_probability_metrics",
                Path(f"output/{test_name}/"),
                force_format="json",
            )
        # elif self.engine_model.__class__.__name__ in [
        #     "LinearDiscriminantAnalysis",
        #     "QuadraticDiscriminantAnalysis",
        # ]:  # If the model is LDA or QDA, calculate the joint log-likelihood and dump it
        #     if self.engine_model.__class__.__name__ == "LinearDiscriminantAnalysis":
        #         joint_log_likelihood_function = lda_joint_loglikelihood

        #     else:
        #         joint_log_likelihood_function = qda_joint_loglikelihood
        #     slogl_data = self._engine_data_preparation(data)
        #     slogl = joint_log_likelihood_function(
        #         self.engine_model,
        #         slogl_data[list(self.data_info["data_columns"])],
        #         slogl_data[TRUE_CLASS_LABEL],
        #     )
        #     prob_metrics_dict = {
        #         "probability_metrics": {"macro avg": {"log_likelihood": slogl}}
        #     }
        #     self._dump_object(
        #         prob_metrics_dict,
        #         self.module_path,
        #         f"{test_name}_probability_metrics",
        #         Path(f"output/{test_name}/"),
        #         force_format="json",
        #     )
        # Dump the scores and generate the metrics
        classes = self._get_model_classes()
        if all(isinstance(c, bool) for c in classes):
            score_columns = ["anomaly_score"]
        else:
            score_columns = [f"{c}_score" for c in classes]

        report_df = self._calculate_metrics(
            data,
            test_name,
            classes=classes,
        )

        self._dump_object(
            data[score_columns],
            self.module_path,
            f"{test_name}_score",
            Path(f"output/{test_name}/"),
        )
        training_time = self.data_info.get(
            "training_time",
            self.train_info.get("training_time", 0),
        )
        time_metrics_dict = {
            "time_metrics": {
                "macro avg": {
                    "training_time": training_time,
                    "testing_time": self.data_info["testing_time"],
                }
            }
        }
        self._dump_object(
            time_metrics_dict,
            self.module_path,
            f"{test_name}_time_metrics",
            Path(f"output/{test_name}/"),
            force_format="json",
        )

        return data

    # TODO: igual esto lo movemos fuera no? En plan a module, para que lo puedan usar el pipeline y demás
    def dump_info_report_assets(self) -> None:
        """Dumps the model, training and data information in a human readable format."""

        # Si no tiene donde guardar... Hacemos un pan con unas tortas
        # O sea, salvo que las tortas sean un argumento module_path en la función,
        # que entonces sí que se puede hacer un pan con unas tortas
        # Bueno, por ahora dejo este assert, pero igual lo quito en el futuro
        assert self.module_path is not None

        # Creates a human readable brief of the model and training information
        report_dict = {
            "Model information": self.model_info,
            "Training information": self.train_info,
            "Data information": self.data_info,
            "Model details": str(self.engine_model),
        }
        self.generate_report(report_dict)

        # RFE: se podría hacer un método "save attributes" que recorra el diccionario de atributos de la clase y los guarde en el path?
        # RFE: igual para esto podemos hacer una lista de los atributos que son salvables y recuperables y aquellos que no (no creo que se pueda guardar y cargar así un modelo de torch)
        # Checks if the model has a scaler and dumps it
        if self.scaler:
            self._dump_object(self.scaler, self.module_path, "scaler")
        # Checks if the model has a scaler and dumps it
        if "vocabulary" in self.__dict__:
            self._dump_object(self.vocabulary, self.module_path, "vocabulary")

    def save_module(
        self,
        module_path: Path = Path(),
        is_full_path: bool = False,
    ) -> None:
        """Overrides the save_module method from the Module class to save the data information.

        Args:
            module_path (str, optional): Path where the model is to be saved. Defaults to an empty string (root path).
            is_full_path (bool, optional): Flag to indicate if the full path is provided. Defaults to False.
            verbose (bool, optional): Flag to indicate if the method should print messages. Defaults to True.
        """
        super().save_module(module_path, is_full_path)
        self._dump_object(self.data_info, self.module_path, "data_info")

    def load_module(self, module_path: Path, is_full_path: bool = False) -> None:
        """Overrides the load_module method from the Module class to load the data information.

        Args:
            module_path (str): Base path where the composite module is saved.
            is_full_path (bool, optional): Flag to indicate if the full path is provided. Defaults to False.
            This flag is used to indicate if the module ID should be appended to the module path, and it is needed for
            every submodule in a modular structure stemming from a top-level composite module. In practice it is used
            for every module and composite module that is not the top-level pipeline.

        """
        super().load_module(module_path, is_full_path)
        # Loads the data information of the model (wrapper generic)
        self.data_info = self._load_object("data_info", "json", self.module_path)
        # Loads the scaler of the model (wrapper generic)
        if "scaling" in self.train_info and self.train_info["scaling"]:
            self.scaler = self._load_object("scaler", "pkl", self.module_path)
        if "trained" in self.train_info and self.train_info["trained"]:
            self.trained = True

    def _make_scaler(self, scaling: str = "") -> None:
        """Create a scaler object based on the scaling method.

        Args:
            scaling (str, optional): Scaling method. Defaults to "".

        Raises:
            ValueError: Scaler file not found or scaling method not implemented
        """
        self.scaler = None
        if scaling:
            scaling = scaling.lower()
            match scaling:
                case "standard":
                    # New standard scaler
                    self.scaler = StandardScaler()
                case "minmax":
                    # New minmax scaler
                    self.scaler = MinMaxScaler()
                case "discretizer":
                    self.scaler = KBinsDiscretizer(
                        n_bins=5, encode="ordinal", strategy="quantile", random_state=0
                    )
                case _:
                    raise ValueError(
                        f"Scaler file not found or scaling method {scaling} not implemented"
                    )

    def _scale_data(
        self, data: pd.DataFrame, should_fit_scaler: bool = False
    ) -> pd.DataFrame:
        """Scale the data using the scaler attribute.

        Args:
            data (pd.DataFrame): Data
            should_fit_scaler (bool, optional): Whether the scaler should be fitted to the data. Defaults to False.

        Raises:
            ValueError: Scaling method not implemented. Scaler must have a transform method

        Returns:
            pd.DataFrame: Scaled data in float32 format
        """
        if self.scaler:
            if not hasattr(self.scaler, "transform"):
                raise ValueError(
                    "Scaling method not implemented. Scaler must have a transform method"
                )
            elif should_fit_scaler:
                # Fit the scaler to the data and scale the data
                scaled_data = pd.DataFrame(
                    self.scaler.fit_transform(data),
                    columns=data.columns,
                    index=data.index,
                )
            else:
                # Only scale the data
                # NOTE: This is done to have retrocompatibility with old scalers
                if any(self.scaler.feature_names_in_ != data.columns):
                    self.log_note(
                        self.level,
                        "The scaler feature names do not match the data columns. Updating scaler feature names.",
                        "WARNING",
                    )
                    self.scaler.feature_names_in_ = data.columns

                scaled_data = pd.DataFrame(
                    np.array(self.scaler.transform(data)),
                    columns=data.columns,
                    index=data.index,
                )

        else:
            scaled_data = data
        # If the scaler is KBinsDiscretizer, ensure the output is integer
        if isinstance(self.scaler, KBinsDiscretizer):
            # Replace numeric bins with string categories
            bin_edges = self.scaler.bin_edges_
            for i, col in enumerate(data.columns):
                bins = bin_edges[i]
                labels = [
                    f"[{bins[j]:.2f}, {bins[j+1]:.2f})" for j in range(len(bins) - 1)
                ]
                scaled_data[col] = pd.cut(
                    scaled_data[col], bins=bins, labels=labels, include_lowest=True
                )
            # Handle NaN values caused by removed bins
            scaled_data = scaled_data.fillna("[Invalid Bin]").astype("category")
        else:
            # Convert the scaled data to float32
            scaled_data = scaled_data.astype("float32")
        return scaled_data

    @staticmethod
    def decode_projection(data, dataset_name, config_dict):
        """
        Decode and return the appropriate column projection based on configuration.
        Determines which columns to use from the dataset by checking projection
        configurations in a prioritized order. Falls back to default behavior if
        no explicit projection is specified.
        Args:
            data: DataFrame-like object containing the dataset with columns and optional attrs.
            dataset_name (str): Name of the dataset for formatting purposes.
            config_dict (dict): Configuration dictionary that may contain:
                - "projection": List of column names to use (formatted via dataset_formatter).
                - "custom_projection": Custom list of column names to use.
                - "exclude_projection": List of column names to exclude from output.
        Returns:
            list: Column names to use for the projection, determined by:
                1. Formatted projection from config if "projection" exists.
                2. Custom projection from config if "custom_projection" exists.
                3. All columns except excluded ones if "exclude_projection" exists.
                4. Embedding columns from data.attrs if "embedding_cols" exists.
                5. All columns except TRUE_CLASS_LABEL and TRUE_ANOMALY_LABEL by default.
        Note:
            TRUE_CLASS_LABEL and TRUE_ANOMALY_LABEL are always excluded from the
            final projection when using exclude_projection or default behavior.
        """

        # Check each projection type in order
        # if config_dict.get("projection") is not None:
        #     return dataset_formatter(dataset_name, config_dict["projection"])

        if config_dict.get("custom_projection") is not None:
            return config_dict["custom_projection"]

        if config_dict.get("exclude_projection") is not None:
            excluded = set(
                config_dict["exclude_projection"]
                + [TRUE_CLASS_LABEL, TRUE_ANOMALY_LABEL]
            )
            return [col for col in data.columns if col not in excluded]

        # Default case
        if "embedding_cols" in data.attrs:
            return data.attrs["embedding_cols"]

        return [
            col
            for col in data.columns
            if col not in {TRUE_CLASS_LABEL, TRUE_ANOMALY_LABEL}
        ]

    # region: Learning methods
    def _before_engine_learn(self, dataset_name: str, data: pd.DataFrame) -> None:
        """Starts the learning process of the model. It is executed before the _engine_learn method,
        and contains the necessary steps to prepare any model for training. This includes updating
        the model information with the recording information and the time at which the training started.

        Args:
            dataset_name (str): Name of the dataset.

        """
        # Updating data information with recording information
        self.data_info["dataset_name"] = dataset_name

        self.data_info["data_columns"] = self.decode_projection(
            data, dataset_name, self.config_dict
        )

        # # If a projection was specified, select the columns of the data that are going to be used for training
        # if "projection" in self.config_dict:
        #     self.data_info["data_columns"] = dataset_formatter(
        #         dataset_name, self.config_dict["projection"]
        #     )
        # elif "custom_projection" in self.config_dict:
        #     # If a custom projection was specified, set it as the columns of the data that are going to be used for training/inference
        #     self.data_info["data_columns"] = self.config_dict["custom_projection"]
        # elif "exclude_projection" in self.config_dict:
        #     # If an excluding projection was specified, exclude such columns from the data that is going to be used for training/inference
        #     self.data_info["data_columns"] = [
        #         col
        #         for col in data.columns
        #         if col
        #         not in self.config_dict["exclude_projection"]
        #         + [TRUE_CLASS_LABEL, TRUE_ANOMALY_LABEL]
        #     ]
        # else:
        #     if "embedding_cols" in data.attrs:
        #         # If the data comes from an embedding, use the embedding cols as the projection
        #         self.data_info["data_columns"] = data.attrs["embedding_cols"]
        #     else:
        #         # If not, take every column except the label columns
        #         self.data_info["data_columns"] = [
        #             col
        #             for col in data.columns
        #             if col not in {TRUE_CLASS_LABEL, TRUE_ANOMALY_LABEL}
        #         ]
        self.data_info["training_time"] = time.time()

    def _engine_data_preparation(
        self, data: pd.DataFrame, routing_column: str | None = None
    ) -> pd.DataFrame:
        """Common data preparation steps for the engine model.

        Args:
            data (pd.DataFrame): Data to prepare.
            routing_column (str | None): Optional routing column for data preparation.
        """
        projection = self.config_dict.get("projection")
        # TODO: move the following 3 lines into moe_model ASAP
        if projection is not None and routing_column is not None:
            protocol_name = projection.upper()
            data = data.loc[data[routing_column] == protocol_name]
        # Apply data budget row limiting if specified in config
        if "data_budget" in self.config_dict:
            data_budget = self.config_dict["data_budget"]
            rows_needed = data_budget.get("rows_needed")

            if rows_needed is not None and len(data) > rows_needed:
                # Limit data to rows_needed if we have more data than needed
                original_rows = len(data)
                data = data.head(rows_needed)
                print(
                    f"Applied data budget: Limited data from {original_rows:,} to {len(data):,} rows (budget: {rows_needed:,})"
                )
            elif rows_needed is not None:
                print(
                    f"Data budget check: Using all {len(data):,} rows (budget allows {rows_needed:,})"
                )

        X_data = data[list(self.data_info["data_columns"])]
        # Define the scaler and store it as a Wrapper attribute
        self._make_scaler(
            self.train_info["scaling"] if "scaling" in self.train_info else ""
        )
        # Scale the training data
        X_data = self._scale_data(X_data, should_fit_scaler=True)
        return X_data

    @abstractmethod
    def _engine_learn(self, data: pd.DataFrame) -> None:
        """Trains the model with the data and the projections.

        Args:
            data (pd.DataFrame): Data to train the model with.

        """
        pass

    def _after_engine_learn(self):
        """Finishes the learning process of the model. It is executed after the _engine_learn method,
        and contains the necessary steps to finalize the training of any model. This includes updating
        the training information with the time it took to train the model and setting the trained attribute
        to True.

        """
        self.data_info["training_time"] = round(
            time.time() - self.data_info["training_time"], 3
        )

        self.data_info["classes"] = self._get_model_classes()
        self.data_info["n_classes"] = len(self.data_info["classes"])
        self.train_info["trained"] = True

    @final
    # Final method that cannot be overridden by child classes
    def _on_learn(
        self, dataset_name: str, data: pd.DataFrame, routing_column: str | None = None
    ) -> None:
        """Learns the model with the data and the projections.

        Args:
            dataset_name (str): Name of the dataset to train the model with.
            data (pd.DataFrame): Data to train the model with.
        """
        self._before_engine_learn(dataset_name, data)
        prepared_data = self._engine_data_preparation(data, routing_column)
        self._engine_learn(prepared_data)
        self._after_engine_learn()

    # endregion: Learning methods


class ClassifierSingleModel(SingleModel):
    true_label = TRUE_CLASS_LABEL

    # TODO: arreglar el lío de que las importancias vienen como una lista de tuplas para poder hacer el variable graph...
    # (Hackazo que metí para la demo)
    def score_instance(self, data: pd.DataFrame) -> tuple[pd.Series, dict]:
        """Scores a single instance of data using the model.

        Args:
            data (pd.DataFrame): The data to be scored.
            dict: empty report dict

        Returns:
            pd.Series: The anomaly score of the instance.
            np.array[float]: The importances of the features in the instance.
        """
        data = self._scale_data(data)
        score = self.engine_model.predict_proba(data)[0]

        return score, {}

    def score_batch(self, data: pd.DataFrame, batch_size: int = 0) -> pd.DataFrame:
        """Score a batch and return class probability columns.

        Column names are derived from ``self._get_model_classes()`` so that
        wrappers using class-label overrides (for example XGBoost with encoded
        targets) still expose human-readable score columns.

        Args:
            data (pd.DataFrame): Input feature matrix.
            batch_size (int, optional): Unused in this implementation. Kept for
                interface compatibility.

        Returns:
            pd.DataFrame: Posterior probabilities with ``<class>_score`` column
            names, or ``anomaly_score`` for boolean/anomaly-style classes.
        """
        data = super().score_batch(data, batch_size)
        classes = self._get_model_classes()
        if all(isinstance(c, bool) for c in classes):
            score_columns = ["anomaly_score"]
        else:
            score_columns = [f"{c}_score" for c in classes]
        score = pd.DataFrame(
            self.engine_model.predict_proba(data),
            index=data.index,
            columns=score_columns,
        )
        return score

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict class labels for the provided data.

        For models that expose numeric class codes at prediction time (notably
        XGBClassifier in this pipeline), this method remaps those codes back to
        the original string labels when ``self._model_classes_override`` is
        populated during training.

        Args:
            data (pd.DataFrame): Input feature matrix.

        Returns:
            pd.Series: Predicted labels aligned with ``data.index``.
        """
        predicted_series = pd.Series(self.engine_model.predict(data), index=data.index)

        if len(self._model_classes_override) > 0 and pd.api.types.is_numeric_dtype(
            predicted_series
        ):
            predicted_codes = predicted_series.astype(int).to_numpy()
            if (
                predicted_codes.size > 0
                and predicted_codes.min() >= 0
                and predicted_codes.max() < len(self._model_classes_override)
            ):
                predicted_series = pd.Series(
                    np.asarray(self._model_classes_override, dtype=object)[
                        predicted_codes
                    ],
                    index=data.index,
                )

        return predicted_series

    def _engine_data_preparation(
        self, data: pd.DataFrame, routing_column: str | None = None
    ) -> pd.DataFrame:
        """Common data preparation steps for classifier models.

        Args:
            data (pd.DataFrame): Data to prepare.

        """
        X_data = super()._engine_data_preparation(
            data[list(data.columns.difference([self.true_label]))], routing_column
        )
        y_data = (
            data.loc[data.index.isin(X_data.index), self.true_label]
            .astype(str)
            .astype("category")
        )
        prepared_data = pd.concat([X_data, y_data], axis=1)
        self.data_info["data_shape"] = prepared_data.shape

        return prepared_data

    def _engine_learn(
        self,
        data: pd.DataFrame,
    ) -> None:
        """Train the wrapped classifier model.

        This method re-initializes ``self.engine_model`` from filtered
        ``model_info`` and fits it with prepared ``X`` and ``y`` data.

        For ``XGBClassifier``, targets are converted from category labels to
        numeric codes for fitting. The original category labels are stored in
        ``self._model_classes_override`` so downstream methods can keep using
        semantic class names.

        Args:
            data (pd.DataFrame): Training data containing features and target
                label column.

        Returns:
            None
        """

        if self.engine_model_class is None:
            raise ValueError("engine_model_class is not set.")
        # Filter model_info to only include arguments that are in the engine_model_class __init__
        model_info_filtered = {
            k: v
            for k, v in self.model_info.items()
            if k in self.engine_model_class.__init__.__code__.co_varnames
        }
        # NOTE: Reinitialization of the engine model with the filtered model_info
        self.engine_model = self.engine_model_class(**model_info_filtered)

        X = data[self.data_info["data_columns"]]
        y = data[self.true_label]
        # NOTE: XGBoost requires numeric labels for fitting.
        # Keep original category labels in a wrapper-level override so the rest
        # of the pipeline can continue working with string class names.
        if self.engine_model_class.__name__ == "XGBClassifier":
            self._model_classes_override = np.asarray(y.cat.categories, dtype=object)
            y = y.cat.codes

        self.engine_model.fit(X, y)
