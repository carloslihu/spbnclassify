from abc import abstractmethod
from pathlib import Path

import pandas as pd

from ...module import Module
from ..evaluator.metric_set import LabelMetricSet


class Thresholder(Module):
    """
    Abstract base class for threshold-based model evaluation and prediction modules.
    The Thresholder class provides a framework for evaluating model predictions using various
    metric sets and applying threshold functions to generate binary or multi-class predictions
    from continuous scores.
    Attributes:
        metric_set_class (type): The class used for storing and computing evaluation metrics.
            Defaults to LabelMetricSet.
        true_label (str): Column name containing true labels in the input data.
        prediction_label (str): Column name for storing predicted labels after thresholding.
        score_columns (list): List of column names containing model scores to threshold.
        thresholder_parameters (dict): Dictionary storing threshold-specific parameters
            that are persisted during save/load operations.
    Note:
        - The class variables (metric_set_class, true_label, prediction_label, score_columns)
          should be reviewed to determine if they should be instance variables initialized
          in __init__ to avoid state sharing issues in interactive environments.
        - The thresholder_parameters attribute should ideally be generalized as "module_parameters"
          at the Module level alongside aggregator parameters.
        - score_columns are recalculated in transform_instance for every instance transformation,
          which could be optimized by calculating once during initialization or fit.
    """

    # TODO: revisar si estas variables deberían ser de clase o de instancia inicializadas en el init
    # (por ejemplo al usarlo dentro de un ipynb se nota la diferencia)
    metric_set_class = LabelMetricSet
    true_label = ""
    prediction_label = ""
    score_columns = []

    def __init__(self, config_dict: dict, module_id: str, level: int):
        super().__init__(config_dict=config_dict, module_id=module_id, level=level)
        # TODO: cambiar thresholder_parameters a algo genérico como "module_parameters" y que sea parte de Module (también para aggregator parameters)
        self.thresholder_parameters = {}

    def evaluate(
        self, data: pd.DataFrame, test_name: str, batch_size: int = 0
    ) -> pd.DataFrame:
        """
        Args:
            test_name (str): Name of the test dataset part to evaluate the models with
            data (pd.DataFrame): Data to evaluate.

        Returns:
            pd.DataFrame: Data with the score of the models
        """
        data = self.transform_batch(data)
        # dump the labels and generate the metrics
        classes = (
            pd.concat([data[self.true_label], data[self.prediction_label]])
            .dropna()
            .unique()
            .tolist()
        )
        self._calculate_metrics(data, test_name, classes=classes)
        self._dump_object(
            data[[self.prediction_label, self.true_label]],
            self.module_path,
            f"{test_name}_label",
            Path(f"output/{test_name}/"),
        )
        return data

    @abstractmethod
    def fit_transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        pass

    def transform_instance(self, data: pd.DataFrame) -> pd.DataFrame:
        # RFE: no puede estar definido aqui y recalcularse por cada vez que se transforma un instance
        self.score_columns = [col for col in data.columns if col.endswith("_score")]
        prediction_df = self._thresholding_function(data[self.score_columns])
        data = pd.concat([data, prediction_df], axis=1)
        return data

    def save_module(
        self, module_path: Path = Path(), is_full_path: bool = False
    ) -> None:
        super().save_module(module_path, is_full_path)
        self._dump_object(
            self.thresholder_parameters,
            self.module_path,
            "thresholder_parameters",
        )

    def load_module(self, module_path: Path, is_full_path: bool = False) -> None:
        super().load_module(module_path, is_full_path)
        self.thresholder_parameters = self._load_object(
            "thresholder_parameters", "json", self.module_path
        )

    @abstractmethod
    def _thresholding_function(self, score: pd.DataFrame | pd.Series) -> pd.Series:
        pass

    @abstractmethod
    def _calculate_threshold(self, score: pd.DataFrame) -> pd.Series:
        pass
