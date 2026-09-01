from pathlib import Path

import pandas as pd

from .metric import (
    ConfusionMatrixMetric,
    CorrectedConfusionMatrixMetric,
    LabelMetric,
    LabelPlotMetric,
    ROCMetric,
    ScoreMetric,
    ScorePlotMetric,
)


class ScoreMetricSet(ScoreMetric):
    """Class that inherits from MetricSet and that is used to obtain each metric and plot
    related to anomaly tests. This is used both to evaluate individual models in the
    Wrapper classes and at a Combiner level.

    Args:
        data (pd.DataFrame): data used to calculate the metrics
        true_label (str OR bool): depends on if anomaly or classification model and data
        with str for classification (type of attack) and bool for anomaly detection
        predicted label (str or bool): depends on if anomaly or classification model and
        data with str for classification (type of attack) and bool for anomaly detection
        score (list): score provided by a model or combiner that will be assessed
        against different metrics.

    Public Methods:
        compute_metrics: generates all metrics and plot related to anomaly tests.
    """

    metric_class_dict: dict[str, type] = {
        "roc_curve": ROCMetric,
        # "pr_curve": PRMetric,
        # "youden_metric": YoudenMetric,
    }

    def __init__(
        self,
        data: pd.DataFrame,
        store_path: Path,
        true_label: str,
        prediction_label: str,
        score_columns: list[str],
        level: int = 0,
        metric_kwargs: dict[str, dict] | None = None,
    ) -> None:
        """Initializes the ScoreMetricSet object.

        Args:
            data (pd.DataFrame): The data used to calculate the metrics.
            store_path (Path): The path where results will be stored.
            true_label (str): The name of the column in the data that contains the true labels
            prediction_label (str): The name of the column in the data that contains the predicted labels
            score_columns (list[str]): The name of the columns in the data that contains the scores
            classes (list[str], optional): The list of classes. If not provided, it defaults to an empty list and will be inferred from y_true. Defaults to [].
            level (int, optional): The level of the module. Defaults to 0.
            score_columns (list[str]): The name of the column in the data that contains the predicted labels
        """
        super().__init__(
            store_path=store_path,
            y_true=data[true_label],
            y_score=data[score_columns],
            level=level,
        )
        self.prediction_label = prediction_label
        self.score_columns = score_columns
        self.metric_kwargs = metric_kwargs or {}
        # Add the ScorePlotMetric for anomaly detection even when a split contains only normal samples.
        # This is useful for debugging score behavior on "normal-only" test chunks.
        if self.is_anomaly_detection:
            self.metric_class_dict["score_plot"] = ScorePlotMetric
        # For classification, only add score plots in the binary case.
        elif set(self.classes_) == set([True, False]):
            self.metric_class_dict["score_plot"] = ScorePlotMetric

    def calculate(self) -> pd.DataFrame:
        """Calculates the metrics and plots for the ScoreMetricSet.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated metrics and plots.
        """
        # Metrics are only computed after asserting that there are more than one unique value in the true
        # label column (i.e. that there are both positive and negative samples). This avoids errors in the calculation of the binary metrics over validation data with only normal examples
        for metric_name in self.metric_class_dict:
            # Skip the metric if it requires multiple y classes and there is only one unique value in the true label column
            if getattr(
                self.metric_class_dict[metric_name],
                "requires_multiple_y_classes",
                False,
            ):
                if len(self.y_true.unique()) == 1:
                    continue
            kwargs = dict(self.metric_kwargs.get(metric_name, {}))
            try:
                self.result_dict[metric_name] = self.metric_class_dict[metric_name](
                    self.store_path, self.y_true, self.y_score, level=self.level + 1, **kwargs
                )
            except TypeError:
                self.result_dict[metric_name] = self.metric_class_dict[metric_name](
                    self.store_path, self.y_true, self.y_score, level=self.level + 1
                )
            self.result_dict[metric_name].calculate()
            # Add the result to the report only if the metric returned a report
            if not len(self.result_dict[metric_name].report_dict) == 0:
                # We get the report dictionary for each metric
                self.report_dict[metric_name] = self.result_dict[
                    metric_name
                ].report_dict

                self.report_df = pd.concat(
                    [self.report_df, self.result_dict[metric_name].report_df],
                    axis=1,
                )

        return self.report_df


class LabelMetricSet(LabelMetric):

    def __init__(
        self,
        data: pd.DataFrame,
        store_path: Path,
        true_label: str,
        prediction_label: str,
        score_columns: list[str],
        level: int = 0,
        metric_kwargs: dict[str, dict] | None = None,
    ) -> None:
        """Initializes the LabelMetricSet object.

        Args:
            data (pd.DataFrame): The data used to calculate the metrics.
            store_path (Path): The path where results will be stored.
            true_label (str): The name of the column in the data that contains the true labels.
            prediction_label (str): The name of the column in the data that contains the predicted labels.
            score_columns (list[str]): The name of the column in the data that contains the predicted labels.
            cv (int, optional): The number of cross-validation folds. Defaults to 5.
            classes (list[str], optional): The list of classes. If not provided, it defaults to an empty list and will be inferred from y_true. Defaults to [].
        """

        # Must not be declared as a class variable because it is modified in the constructor
        self.metric_class_dict = {
            "confusion_matrix": ConfusionMatrixMetric,
            "label_plot": LabelPlotMetric,
        }

        super().__init__(
            store_path=store_path,
            y_true=data[true_label],
            y_pred=data[prediction_label],
            level=level,
        )
        self.prediction_label = prediction_label
        self.score_columns = score_columns
        self.metric_kwargs = metric_kwargs or {}
        # Add the corrected confusion matrix metric if there are only binary classes (normality/attack)
        if set(self.classes_) == set([True, False]):
            self.metric_class_dict["corrected_confusion_matrix"] = (
                CorrectedConfusionMatrixMetric
            )

    def calculate(self) -> pd.DataFrame:
        """Calculates the metrics and plots for the LabelMetricSet.

        Returns:
            dict: A dictionary containing the calculated metrics and plots.
        """
        for metric_name in self.metric_class_dict:
            # Skip the metric if it requires multiple y classes and there is only one unique value in the true label column
            if getattr(
                self.metric_class_dict[metric_name],
                "requires_multiple_y_classes",
                False,
            ):
                if len(self.y_true.unique()) == 1:
                    continue
            kwargs = dict(self.metric_kwargs.get(metric_name, {}))
            try:
                self.result_dict[metric_name] = self.metric_class_dict[metric_name](
                    self.store_path,
                    self.y_true,
                    self.y_pred,
                    self.level + 1,
                    **kwargs,
                )
            except TypeError:
                self.result_dict[metric_name] = self.metric_class_dict[metric_name](
                    self.store_path, self.y_true, self.y_pred, self.level + 1
                )
            self.result_dict[metric_name].calculate()
            # Add the result to the report only if the metric returned a report
            if not len(self.result_dict[metric_name].report_dict) == 0:
                self.report_dict[metric_name] = self.result_dict[
                    metric_name
                ].report_dict
        return self.report_df
