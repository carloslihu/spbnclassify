import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from ...module import Module

# Ignore division by zero and invalid values
np.seterr(divide="ignore", invalid="ignore")
sns.set_theme()


def set_logging_level(level):
    match level:
        case "DEBUG":
            log_level = logging.DEBUG
        case "INFO":  # DEFAULT
            log_level = logging.INFO
        case "WARNING":
            log_level = logging.WARNING
        case "ERROR":
            log_level = logging.ERROR
        case "CRITICAL":
            log_level = logging.CRITICAL
        case _:
            raise ValueError(f"Logging level {level} not supported.")
    # logger.setLevel(log_level)
    logging.getLogger().setLevel(log_level)


# region Abstract Metric classes
class Metric(ABC):
    def __init__(
        self,
        store_path: Path,
        y_true: pd.Series,
        level: int = 0,
    ) -> None:
        """Initialize the evaluator with the given module path, true labels, and classes.

        Args:
            store_path (Path): The path where results will be stored.
            y_true (pd.Series): The true labels for the evaluation.
            classes (str | list[str], optional): The list of classes or a single class as a string.
            If not provided, it defaults to an empty list and will be inferred from y_true. Defaults to [].
        """
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.y_true = y_true

        # If the task is AD, it does not need string casting. Classification does need this step
        self.is_anomaly_detection = (y_true.dtype == bool) or set(
            y_true.unique()
        ).issubset({0, 1})
        # NOTE: the additional condition "set(y_true.unique())=={0,1}" was added to provide external AD model compatibility
        if not self.is_anomaly_detection:
            self.y_true = self.y_true.astype(str)

        self.level = level
        self.classes_ = sorted(self.y_true.dropna().unique().tolist())

        # Dictionary to store all intermediate results of the metric calculation for each class
        self.result_dict = {}

        # JSON serializable dictionary with results for each class (metrics.json)
        self.report_dict = {}

        # DataFrame to summarize the results of the metric calculation (obtained from report_dict)
        self.report_df = pd.DataFrame()

    @abstractmethod
    def calculate(self) -> pd.DataFrame:  # pragma: no cover
        """Method that calculates the metrics and plots related to the test execution.

        Returns:
            dict: dictionary with the metrics and plots obtained in the test execution.
        """
        pass


class ScoreMetric(Metric):
    def __init__(
        self,
        store_path: Path,
        y_true: pd.Series,
        y_score: pd.DataFrame,
        level: int = 0,
    ) -> None:
        """Initialize the evaluator with the given module path, true labels, and scores.

        Args:
            store_path (Path): The path where results will be stored.
            y_true (pd.Series): The true labels for the evaluation.
            y_score (pd.DataFrame): The scores for the evaluation.
            classes (str | list[str], optional): The list of classes or a single class as a string. Defaults to [].
        """
        super().__init__(store_path, y_true, level)
        self.y_score = y_score


class LabelMetric(Metric):
    def __init__(
        self,
        store_path: Path,
        y_true: pd.Series,
        y_pred: pd.Series,
        level: int = 0,
    ) -> None:
        """Initialize the evaluator with the given module path, true labels, and predictions.

        Args:
            store_path (Path): The path where results will be stored.
            y_true (pd.Series): The true labels for the evaluation.
            y_pred (pd.Series): The predictions for the evaluation.
            level (int, optional): The level of the module. Defaults to 0.
        """
        super().__init__(store_path, y_true, level)
        self.y_pred = y_pred
        self.classes_ = sorted(
            pd.concat([self.y_true, self.y_pred]).dropna().unique().tolist()
        )


# endregion

# region Score Metrics


class ROCMetric(ScoreMetric):

    requires_multiple_y_classes = True

    # RFE: Integrate changes from legacy metrics.py::make_roc_plots like cutoff values
    def __init__(
        self,
        store_path: Path,
        y_true: pd.Series,
        y_score: pd.DataFrame,
        level: int = 0,
        x_scale: str = "linear",
        x_log_min: float = 1e-6,
        symlog_linthresh: float = 1e-3,
    ) -> None:
        super().__init__(
            store_path=store_path, y_true=y_true, y_score=y_score, level=level
        )
        self.x_scale = x_scale
        self.x_log_min = float(x_log_min)
        self.symlog_linthresh = float(symlog_linthresh)

    def calculate(self) -> pd.DataFrame:
        """Method that calculates the metrics and plots related to the test execution.

        Returns:
            dict: dictionary with the metrics and plots obtained in the test execution.
        """

        def calculate_individual_roc_curve(
            y_true: pd.Series,
            y_score: pd.Series,
        ) -> pd.DataFrame:
            """Calculates the Receiver Operating Characteristic (ROC) curve and returns its values.

            Args:
                y_true (np.ndarray): Ground truth (correct) target values.
                y_score (np.ndarray): Target scores (log_likelihoods).

            Returns:
                pd.DataFrame: The ROC curve values
                - threshold
                - fpr: False Positive Rate
                - tpr: True Positive Rate
                - roc_auc: Area Under the ROC curve
            """
            # Compute ROC curve and area
            # Get non-NaN indices from self.y_score
            non_nan_mask = ~y_score.isna()
            y_true = y_true[non_nan_mask]
            y_score = y_score[non_nan_mask]
            if len(y_true.unique()) < 2:
                print(
                    "ROC curve cannot be computed because there is maximum one class present in y_true. Returning default values."
                )
                fpr, tpr, thresholds = (
                    np.array([0.0, 1.0]),
                    np.array([0.0, 1.0]),
                    np.array([np.inf, -np.inf]),
                )
            else:
                fpr, tpr, thresholds = roc_curve(
                    y_true, y_score, drop_intermediate=False
                )
            roc_auc = auc(fpr, tpr)
            roc_df = pd.DataFrame(
                {
                    "threshold": thresholds,
                    "FPR": fpr,
                    "TPR": tpr,
                    "ROC_AUC": roc_auc,
                }
            )
            return roc_df

        # Score might include NaN values at the beginning because of the transformer window
        # Get non-NaN indices from self.y_score
        # In the case of having multiple columns, we will take the first one
        non_nan_mask = ~self.y_score.iloc[:, 0].isna().values.flatten()
        self.y_score = self.y_score[non_nan_mask]
        self.y_true = self.y_true[non_nan_mask]

        if self.is_anomaly_detection:
            available_classes = ["anomaly"]
            y_true = pd.DataFrame({"anomaly": self.y_true})
        else:
            # NOTE: Also see which scores can be calculated (available to be predicted by class)
            available_classes = sorted(
                [
                    class_name
                    for class_name in self.classes_
                    if f"{class_name}_score" in self.y_score.columns
                ]
            )
            y_true_binarized = np.array(
                label_binarize(self.y_true, classes=self.classes_)
            )
            # So that it always returns a 2D array with binary classes
            if y_true_binarized.shape[1] == 1:
                y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])
            y_true = pd.DataFrame(
                y_true_binarized,
                columns=self.classes_,
                index=self.y_score.index,
            )
        if available_classes == []:
            raise ValueError(
                f"No available classes with scores found. Available classes: {self.classes_}, score columns: {self.y_score.columns}"
            )

        # For each available class, calculate the ROC curve
        for class_name in available_classes:
            self.result_dict[class_name] = calculate_individual_roc_curve(
                y_true[class_name], self.y_score[f"{class_name}_score"]
            )
            class_roc_auc = self.result_dict[class_name]["ROC_AUC"].iloc[0]
            self.report_dict.update(
                {
                    class_name: {
                        "ROC_AUC": class_roc_auc,
                        "support": y_true[class_name].sum(),
                    },
                }
            )
        if available_classes != ["anomaly"]:
            self.report_dict.update(
                {
                    "macro avg": {
                        "ROC_AUC": np.nanmean(
                            [
                                self.report_dict[class_name]["ROC_AUC"]
                                for class_name in available_classes
                            ]
                        )
                    },
                    "weighted avg": {
                        "ROC_AUC": np.average(
                            [
                                self.report_dict[class_name]["ROC_AUC"]
                                for class_name in available_classes
                            ],
                            weights=[
                                self.report_dict[class_name]["support"]
                                for class_name in available_classes
                            ],
                        )
                    },
                }
            )
        self.report_df = pd.DataFrame(self.report_dict).transpose()
        self.report_df.index.name = "class"
        self.report_df.reset_index(inplace=True)

        # Save Metrics to CSV and LaTeX
        self.report_df.to_csv(self.store_path / "metrics.csv")
        self.report_df.to_latex(
            self.store_path / "metrics.tex",
            float_format="%.3f",
            caption="ROC metrics report.",
            label="tab:roc_metrics",
            index=False,
            escape=True,
            position="htbp",
        )

        self.save_plot()

        return self.report_df

    def save_plot(
        self,
        alpha_mod: float = 0.5,
        n_plots: int = 0,
    ) -> None:
        """Method that saves the plot generated by the calculate method.

        Args:
            alpha_mod (float, optional): Alpha modifier for the fill_between. Defaults to 0.5.
            n_plots (int, optional): Number of plots. Defaults to 0.
        """
        fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size

        x_scale = (self.x_scale or "linear").lower()
        if x_scale not in {"linear", "log", "symlog"}:
            raise ValueError(
                f"Invalid x_scale={self.x_scale!r}. Supported: 'linear', 'log', 'symlog'."
            )

        for class_name, roc_df in self.result_dict.items():
            fpr = roc_df["FPR"]
            tpr = roc_df["TPR"]
            roc_auc = roc_df["ROC_AUC"].iloc[0]

            # For log-scale plotting, avoid log(0) by clipping only for display.
            if x_scale == "log":
                fpr_plot = np.maximum(fpr.to_numpy(dtype=float), self.x_log_min)
                fpr_plot = pd.Series(fpr_plot, index=fpr.index, name=fpr.name)
            else:
                fpr_plot = fpr

            disp = RocCurveDisplay(
                fpr=fpr_plot,
                tpr=tpr,
                roc_auc=roc_auc,
                name=f"{class_name} ROC curve",
            )

            disp.plot(
                ax=ax,
                plot_chance_level=False,
                label=f"{class_name} (area = {roc_auc:0.3f})",
                linewidth=2,  # Make lines thicker
            )
            # ax.fill_between(
            #     fpr,
            #     tpr,
            #     alpha=0.75 - alpha_mod * n_plots,
            # )

        # Plot No skill line (0,0) -> (1,1)
        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            lw=2,
            linestyle="--",
            label="No Skill",
            alpha=0.8,
        )

        if x_scale == "log":
            ax.set_xscale("log")
            ax.set_xlim(left=self.x_log_min, right=1.0)
        elif x_scale == "symlog":
            ax.set_xscale("symlog", linthresh=self.symlog_linthresh)

        ax.axis("square")
        ax.set_xlabel("False Positive Rate", fontsize=16)
        ax.set_ylabel("True Positive Rate", fontsize=16)
        ax.set_title(
            "Receiver Operating Characteristic Curve\nOne-vs-Rest multiclass",
            fontsize=18,
            pad=20,
        )

        # Improve legend
        legend = ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        # Improve tick labels
        ax.tick_params(axis="both", which="major", labelsize=12)

        # Saving the figure with tight layout and high DPI
        if self.store_path:
            fig.savefig(
                self.store_path / "roc_curve.png",
                bbox_inches="tight",
                dpi=300,
                facecolor="white",
                edgecolor="none",
            )
        # Close the figure to free up memory
        plt.close(fig)


# endregion


# region Label Metrics


class ConfusionMatrixMetric(LabelMetric):

    requires_multiple_y_classes = True

    def calculate(self) -> pd.DataFrame:
        """Method that calculates the metrics and plots related to the test execution.

        Returns:
            dict: dictionary with the metrics and plots obtained in the test execution.
        """
        # Get non-NaN indices from self.y_score
        non_nan_mask = ~self.y_pred.isna().values.flatten()

        # Filter non-NaN indices
        self.y_pred = self.y_pred[non_nan_mask]
        self.y_true = self.y_true[non_nan_mask]
        # NOTE: This is the returned format from scikit-learn in the binary case
        # tn, fp
        # fn, tp
        self.cm = confusion_matrix(self.y_true, self.y_pred, labels=self.classes_)

        self.report_dict.update(
            classification_report(
                self.y_true,
                self.y_pred,
                target_names=self.classes_,
                output_dict=True,
                zero_division=np.nan,
            )
        )
        # If the available classes are boolean (anomaly detection)
        if all(isinstance(item, bool) for item in self.classes_):
            available_classes = ["anomaly"]
            # Remove the extra keys from the report_dict and fill the "anomaly" case
            self.report_dict.pop(False, None)
            self.report_dict.pop("macro avg", None)
            self.report_dict.pop("weighted avg", None)

            self.report_dict["anomaly"] = self.report_dict.pop(True)
            self.report_dict["anomaly"]["accuracy"] = self.report_dict.pop("accuracy")

        # If the available classes are strings (classification)
        else:
            available_classes = sorted(self.classes_)
        # region Class metrics
        # Based on https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
        FP = self.cm.sum(axis=0) - np.diag(self.cm)
        FN = self.cm.sum(axis=1) - np.diag(self.cm)
        TP = np.diag(self.cm)
        TN = self.cm.sum() - (FP + FN + TP)

        P = TP + FN
        N = TN + FP

        for idx, class_name in enumerate(available_classes):
            # We only want the metrics for the anomaly class
            if available_classes == ["anomaly"]:
                idx = 1
            # We follow the calculation order from left to right and up to down found in https://en.wikipedia.org/wiki/Confusion_matrix
            self.report_dict[class_name].update(
                {
                    "TPR": self.report_dict[class_name].pop("recall"),
                    "FNR": FN[idx] / P[idx],  # NaN when "No positive samples"
                    "FPR": FP[idx] / N[idx],  # NaN when "No negative samples"
                    "TNR": TN[idx] / N[idx],  # NaN when "No negative samples"
                    "PPV": self.report_dict[class_name].pop(
                        "precision"
                    ),  # NaN when "No positive predictions"
                    "FOR": FN[idx]
                    / (TN[idx] + FN[idx]),  # NaN when "No negative predictions"
                    "FDR": FP[idx]
                    / (TP[idx] + FP[idx]),  # NaN when "No positive predictions"
                    "NPV": TN[idx]
                    / (TN[idx] + FN[idx]),  # NaN when "No negative predictions"
                    "F1-score": self.report_dict[class_name].pop(
                        "f1-score"
                    ),  # NaN when "No positive samples"
                }
            )
        # endregion Class metrics

        # region Average the metrics
        if available_classes != ["anomaly"]:
            self.report_dict["macro avg"].update(
                {
                    "TPR": self.report_dict["macro avg"].pop("recall"),
                    "FPR": np.nanmean(
                        [
                            self.report_dict[class_name]["FPR"]
                            for class_name in available_classes
                        ]
                    ),
                    "TNR": np.nanmean(
                        [
                            self.report_dict[class_name]["TNR"]
                            for class_name in available_classes
                        ]
                    ),
                    "FNR": np.nanmean(
                        [
                            self.report_dict[class_name]["FNR"]
                            for class_name in available_classes
                        ]
                    ),
                    "PPV": self.report_dict["macro avg"].pop("precision"),
                    "FDR": np.nanmean(
                        [
                            self.report_dict[class_name]["FDR"]
                            for class_name in available_classes
                        ]
                    ),
                    "NPV": np.nanmean(
                        [
                            self.report_dict[class_name]["NPV"]
                            for class_name in available_classes
                        ]
                    ),
                    "FOR": np.nanmean(
                        [
                            self.report_dict[class_name]["FOR"]
                            for class_name in available_classes
                        ]
                    ),
                    "F1-score": self.report_dict["macro avg"].pop("f1-score"),
                    "accuracy": self.report_dict.pop("accuracy"),
                }
            )
            self.report_dict["weighted avg"].update(
                {
                    "TPR": self.report_dict["weighted avg"].pop("recall"),
                    "FPR": np.average(
                        [
                            self.report_dict[class_name]["FPR"]
                            for class_name in available_classes
                        ],
                        weights=[
                            self.report_dict[class_name]["support"]
                            for class_name in available_classes
                        ],
                    ),
                    "TNR": np.average(
                        [
                            self.report_dict[class_name]["TNR"]
                            for class_name in available_classes
                        ],
                        weights=[
                            self.report_dict[class_name]["support"]
                            for class_name in available_classes
                        ],
                    ),
                    "FNR": np.average(
                        [
                            self.report_dict[class_name]["FNR"]
                            for class_name in available_classes
                        ],
                        weights=[
                            self.report_dict[class_name]["support"]
                            for class_name in available_classes
                        ],
                    ),
                    "FDR": np.average(
                        [
                            self.report_dict[class_name]["FDR"]
                            for class_name in available_classes
                        ],
                        weights=[
                            self.report_dict[class_name]["support"]
                            for class_name in available_classes
                        ],
                    ),
                    "NPV": np.average(
                        [
                            self.report_dict[class_name]["NPV"]
                            for class_name in available_classes
                        ],
                        weights=[
                            self.report_dict[class_name]["support"]
                            for class_name in available_classes
                        ],
                    ),
                    "FOR": np.average(
                        [
                            self.report_dict[class_name]["FOR"]
                            for class_name in available_classes
                        ],
                        weights=[
                            self.report_dict[class_name]["support"]
                            for class_name in available_classes
                        ],
                    ),
                    "PPV": self.report_dict["weighted avg"].pop("precision"),
                    "F1-score": self.report_dict["weighted avg"].pop("f1-score"),
                    "accuracy": self.report_dict["macro avg"]["accuracy"],
                }
            )
        # cr_df = pd.DataFrame(cr_dict).transpose()
        # endregion Average metrics
        self.report_df = pd.DataFrame(self.report_dict).transpose()
        self.report_df.index.name = "class"
        self.report_df.reset_index(inplace=True)

        # Save Metrics to CSV and LaTeX
        self.report_df.to_csv(self.store_path / "metrics.csv")
        self.report_df.to_latex(
            self.store_path / "metrics.tex",
            float_format="%.3f",
            caption="Confusion matrix metrics report.",
            label="tab:confusion_matrix_metrics",
            index=False,
            escape=True,
            position="htbp",
        )

        self.save_plot()
        return self.report_df

    def save_plot(self):
        """Method that saves the plot generated by the calculate method."""
        num_classes = len(self.classes_)
        fig_size = max(8, num_classes)  # Ensure a minimum size of 8
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))  # Adjust figure size
        ax.set_title("Confusion Matrix", fontsize=16)  # Increase title font size
        ax.grid(False)  # Disable grid lines
        # normalized_confusion_matrix = (
        #     self.cm.astype("float") / self.cm.sum(axis=1)[:, np.newaxis]
        # )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=self.cm, display_labels=self.classes_
        )
        disp.plot(
            ax=ax, xticks_rotation="vertical", cmap="Blues"
        )  # Use a colormap, format values, and normalize
        plt.xticks(fontsize=12)  # Increase x-axis label font size
        plt.yticks(fontsize=12)  # Increase y-axis label font size
        plt.xlabel("Predicted Label", fontsize=14)  # Increase x-axis title font size
        plt.ylabel("True Label", fontsize=14)  # Increase y-axis title font size
        fig.savefig(
            self.store_path / "confusion_matrix.png", bbox_inches="tight", dpi=300
        )
        # RFE: Add to mlflow each confusion matrix for each test
        # mlflow.log_figure(
        #     figure=fig,
        #     artifact_file=str(self.store_path / "confusion_matrix.png"),
        # )
        plt.close(fig)


# TODO: unused... Remove? Could be useful someday
# # Experimental class: the Confusion Matrix is corrected so that a positive prediction is not considered false if it lies within the window_size of a true positive
# class WindowedConfusionMatrixMetric(LabelMetric):

#     def calculate(self) -> pd.DataFrame:
#         """Method that calculates the metrics and plots related to the test execution.

#         Returns:
#             dict: dictionary with the metrics and plots obtained in the test execution.
#         """
#         # Get non-NaN indices from self.y_score
#         non_nan_mask = ~self.y_pred.isna().values.flatten()

#         # Filter non-NaN indices
#         self.y_pred = self.y_pred[non_nan_mask]
#         self.y_true = self.y_true[non_nan_mask]

#         # Get window_size metadata
#         window_size = self.y_true.attrs.get("window_size")
#         # Find true positive indices
#         indices = np.where(self.y_true)[0]
#         # Expand indices to window_size
#         for index in indices:
#             self.y_true[index : index + window_size] = True

#         self.wcm = confusion_matrix(
#             self.y_true, self.y_pred, labels=self.classes_, normalize="true"
#         )

#         wcm_dict = {}
#         # TODO: hacer esto pero para multiclass... Que actualice ratios 1vsALL para cada clase
#         if len(self.classes_) == 2:
#             wcm_dict["accuracy"] = sum(np.diag(self.wcm)) / self.wcm.sum()
#             if sum(self.wcm[0, :]) == 0:
#                 wcm_dict["TPR"] = "No positive samples"
#                 wcm_dict["FPR"] = "No positive samples"
#             else:
#                 wcm_dict["TPR"] = self.wcm[0, 1] / sum(self.wcm[0, :])
#                 wcm_dict["FPR"] = self.wcm[0, 0] / sum(self.wcm[0, :])
#             if sum(self.wcm[1, :]) == 0:
#                 wcm_dict["TNR"] = "No negative samples"
#                 wcm_dict["FNR"] = "No negative samples"
#             else:
#                 wcm_dict["TNR"] = self.wcm[1, 1] / sum(self.wcm[1, :])
#                 wcm_dict["FNR"] = self.wcm[1, 0] / sum(self.wcm[1, :])
#             if sum(self.wcm[:, 0]) == 0:
#                 wcm_dict["PPV"] = "No positive predictions"
#                 wcm_dict["FDR"] = "No positive predictions"
#             else:
#                 wcm_dict["PPV"] = self.wcm[0, 0] / sum(self.wcm[:, 0])
#                 wcm_dict["FDR"] = self.wcm[1, 0] / sum(self.wcm[:, 0])
#             if sum(self.wcm[:, 1]) == 0:
#                 wcm_dict["NPV"] = "No negative predictions"
#                 wcm_dict["FOR"] = "No positive predictions"
#             else:
#                 wcm_dict["NPV"] = self.wcm[1, 1] / sum(self.wcm[:, 1])
#                 wcm_dict["FOR"] = self.wcm[0, 1] / sum(self.wcm[:, 1])
#             self.report_dict.update({f"binary_wcm": wcm_dict})

#         self.save_plot()
#         return self.report_df

#     def save_plot(self):
#         """Method that saves the plot generated by the calculate method."""
#         fig, ax = plt.subplots(layout="tight")
#         ax.set_title("Windowed Confusion Matrix")
#         ax.grid(False)  # Disable grid lines
#         disp = ConfusionMatrixDisplay(
#             confusion_matrix=self.wcm, display_labels=self.classes_
#         )
#         disp.plot(ax=ax, xticks_rotation="vertical")
#         fig.savefig(self.store_path / "windowed_confusion_matrix.png")
#         # RFE: Add to mlflow
#         # mlflow.log_artifact(str(self.store_path / "windowed_confusion_matrix.png"))
#         plt.close(fig)


# Experimental class: the Confusion Matrix is corrected so that a positive prediction is not considered false if it lies in the range between the first and last true labels, extended by window_size at its end
class CorrectedConfusionMatrixMetric(LabelMetric):

    def calculate(self) -> pd.DataFrame:
        """Method that calculates the metrics and plots related to the test execution.

        Returns:
            dict: dictionary with the metrics and plots obtained in the test execution.
        """
        if any(self.y_pred.isna()):
            Module.log_note(
                self.level,
                f"NaN values detected in the predictions. They will be removed to form the evaluation set.",
                category="WARNING",
            )
            # Get non-NaN indices from self.y_score
            non_nan_mask = ~self.y_pred.isna().values.flatten()
            # Filter non-NaN indices
            self.y_pred = self.y_pred[non_nan_mask]
            self.y_true = self.y_true[non_nan_mask]

        # Get window_size metadata
        if "window_size" in self.y_true.attrs:
            window_size = self.y_true.attrs["window_size"]
        else:
            window_size = 1
        # Find the first true positive index
        first_true_index = np.argmax(self.y_true)
        # Find the last true positive index
        last_true_index = len(self.y_true) - 1 - np.argmax(self.y_true[::-1])

        # Define the normality period (safe period of normality BEFORE the attack)
        # Data after the attack is executed is not considered safe, since aggregators might continue accumulating score
        self.normality_true = pd.concat([self.y_true[: first_true_index - 1]])
        self.normality_pred = pd.concat([self.y_pred[: first_true_index - 1]])
        self.normality_len = len(self.normality_true)

        # Define the attack period
        # Note that the attack period is extended by window_size at its end, with a +1 needed to include the last true positive (according to pandas slicing methods)
        self.attack_true = self.y_true[
            first_true_index : last_true_index + window_size + 1
        ]
        self.attack_pred = self.y_pred[
            first_true_index : last_true_index + window_size + 1
        ]
        self.attack_len = len(self.attack_true)

        # Compute the confusion matrix for the normality period
        self.normality_cm = confusion_matrix(
            self.normality_true,
            self.normality_pred,
            labels=self.classes_,
            normalize="true",
        )

        # Compute the confusion matrix for the attack period
        self.attack_cm = confusion_matrix(
            np.ones(len(self.attack_true)),
            self.attack_pred,
            labels=self.classes_,
            normalize="true",
        )

        # NOTE: periods of normality enclosing the attack are used to compute FPR
        # NOTE: the attack period is used to compute TPR (should consider length of period, ratio of attack flows to total flows in the attack period and window length?)
        # NOTE: TNR is not important, really
        # NOTE: FNR is replaced by a measure considering the number of false negatives against the ratio of the attack period to the whole period, as well as the number of attack flows (and window_length) to the total number of flows in the attack period (although some of this flows may not be normal, but a response to an attack)

        ccm_dict = {}
        # TODO: hacer esto pero para multiclass... Que actualice ratios 1vsALL para cada clase
        if len(self.classes_) == 2:
            # Assert the classes are sorted (0/1 or False/True)
            assert self.classes_ == sorted(self.classes_)
            # The accuracy is calculated over the whole period (discarding the last normality period of aggregator saturation)
            ccm_dict["accuracy"] = (
                sum(np.diag(self.attack_cm)) + sum(np.diag(self.normality_cm))
            ) / (self.attack_cm.sum() + self.normality_cm.sum())
            # Measures of how many positive samples are detected are extracted ONLY fom the attack period
            if sum(self.attack_cm[1, :]) == 0:
                ccm_dict["TPR"] = np.nan  # "No positive samples"
                ccm_dict["FNR"] = np.nan  # "No positive samples"
            else:
                ccm_dict["TPR"] = self.attack_cm[1, 1] / sum(self.attack_cm[1, :])
                ccm_dict["FNR"] = self.attack_cm[1, 0] / sum(self.attack_cm[1, :])
            # Measures of how many negative samples were incorrectly detected are extracted ONLY fom the normality period(discarding the last normality period of aggregator saturation)
            if sum(self.normality_cm[0, :]) == 0:
                ccm_dict["TNR"] = np.nan  # "No negative samples"
                ccm_dict["FPR"] = np.nan  # "No negative samples"
            else:
                ccm_dict["TNR"] = self.normality_cm[0, 0] / sum(self.normality_cm[0, :])
                ccm_dict["FPR"] = self.normality_cm[0, 1] / sum(self.normality_cm[0, :])
            # Measures of how trustworthy the positive predictions are come extracted from the whole period (discarding the last normality period of aggregator saturation)
            if sum(self.attack_cm[:, 1]) == 0:
                ccm_dict["PPV"] = np.nan  # "No positive predictions"
                ccm_dict["FDR"] = np.nan  # "No positive predictions"
            else:
                ccm_dict["PPV"] = (self.attack_cm[1, 1] + self.normality_cm[1, 1]) / (
                    sum(self.attack_cm[:, 1]) + sum(self.normality_cm[:, 1])
                )
                ccm_dict["FDR"] = (self.attack_cm[0, 1] + self.normality_cm[0, 1]) / (
                    sum(self.attack_cm[:, 1]) + sum(self.normality_cm[:, 1])
                )
            # Measures of how trustworthy the negative predictions are come extracted ONLY fom the normality period (discarding the last normality period of aggregator saturation)
            if sum(self.normality_cm[:, 0]) == 0:
                ccm_dict["NPV"] = np.nan  # "No negative predictions"
                ccm_dict["FOR"] = np.nan  # "No negative predictions"
            else:
                ccm_dict["NPV"] = self.normality_cm[0, 0] / sum(self.normality_cm[:, 0])
                ccm_dict["FOR"] = self.normality_cm[1, 0] / sum(self.normality_cm[:, 0])

            # Simplified measures for quick report
            if ccm_dict["TPR"] > 0:
                ccm_dict["discovery"] = True
            else:
                ccm_dict["discovery"] = False

            # Computation of the number of alarm tracks (tracks of ones) in the normality period
            add_tracks = 0
            if self.normality_pred.iloc[0] or self.normality_pred.iloc[-1]:
                add_tracks = 1
            ccm_dict["alarm_tracks"] = (
                int(sum(abs(np.diff(self.normality_pred))) / 2) + add_tracks
            )

            self.report_dict = ccm_dict

        self.save_plot()
        return self.report_df

    def save_plot(self):
        """Method that saves the plot generated by the calculate method."""
        fig, ax = plt.subplots(layout="tight")
        ax.set_title("Attack Confusion Matrix")
        ax.grid(False)  # Disable grid lines
        disp = ConfusionMatrixDisplay(
            confusion_matrix=self.attack_cm, display_labels=self.classes_
        )
        disp.plot(ax=ax, xticks_rotation="vertical")
        fig.savefig(self.store_path / "attack_confusion_matrix.png")
        # RFE: Add to mlflow
        # mlflow.log_artifact(str(self.store_path / "attack_confusion_matrix.png"))
        plt.close(fig)

        fig, ax = plt.subplots(layout="tight")
        ax.set_title("Normality Confusion Matrix")
        ax.grid(False)  # Disable grid lines
        disp = ConfusionMatrixDisplay(
            confusion_matrix=self.normality_cm, display_labels=self.classes_
        )
        disp.plot(ax=ax, xticks_rotation="vertical")
        fig.savefig(self.store_path / "normality_confusion_matrix.png")
        # RFE: Add to mlflow
        # mlflow.log_artifact(str(self.store_path / "normality_confusion_matrix.png"))
        plt.close(fig)


class LabelPlotMetric(LabelMetric):

    def calculate(self) -> pd.DataFrame:
        """Method that calculates the metrics and plots related to the test execution."""
        # Get non-NaN indices from self.y_score
        non_nan_mask = ~self.y_pred.isna().values.flatten()

        # Filter non-NaN indices
        self.y_pred = self.y_pred[non_nan_mask]
        self.y_true = self.y_true[non_nan_mask]

        try:
            self.save_plot()
        except Exception as e:
            Module.log_note(
                self.level,
                f"An error occurred while saving the plot: {e}",
                category="ERROR",
            )
        return self.report_df

    def make_plot(self):
        """Method that saves the plot generated by the calculate method."""
        plot_data = pd.concat([self.y_true, self.y_pred], axis=1).reset_index()
        true_name = str(self.y_true.name)
        pred_name = str(self.y_pred.name)
        plot_data[true_name] = plot_data[true_name].astype(str)
        plot_data[pred_name] = plot_data[pred_name].astype(str)
        set_logging_level("ERROR")
        g = sns.catplot(
            data=plot_data,
            x="index",
            y=true_name,
            dodge=True,
            aspect=3,
            height=4,
            hue=pred_name,
        ).set(title=f"Category {pred_name} plot\n")
        set_logging_level("INFO")
        for i in range(plot_data[true_name].nunique() - 1):
            g.ax.axhline(i + 0.5, color="white", alpha=1)
        g.add_legend()  # Ensure the legend is added to the FacetGrid
        sns.move_legend(
            g,
            "upper right",
            bbox_to_anchor=(0.975, 0.975),
            ncol=plot_data[pred_name].nunique(),
            title="",
            frameon=False,
        )
        plt.tight_layout(pad=1)

    def show_plot(self):
        """Method that shows the plot generated by the calculate method."""
        self.make_plot()
        plt.show()

    def save_plot(self):
        """Method that saves the plot generated by the calculate method."""
        self.make_plot()
        plt.savefig(self.store_path / "alarm_plot.png")
        plt.close()


# endregion

# class PRMetric(ScoreMetric):
#     def __init__(
#         self,
#         data: pd.DataFrame,
#         true_label: str,
#         pred_columns: list,
#     ):
#         super().__init__(data, true_label, pred_columns)
#         self.y_score = score

#     def calculate(self):
#         nans = sum(np.isnan(x) for x in self.y_score)
#         metric_score = self.y_score[nans:]
#         y_true = self.data[self.true_label][nans:]
#         self.precision, self.recall, self.thresholds = precision_recall_curve(
#             y_true,
#             metric_score,
#         )
#         pr_auc = auc(self.recall, self.precision)
#         # NOTE: Last precision and recall values are 1 and 0 respectively
#         # NOTE: So we have to adapt the thresholds to be the same length as precision and recall
#         self.thresholds = np.append(self.thresholds, self.thresholds[-1] + 1)
#         self.report_dict = {
#             "pr_auc": pr_auc,
#         }
#         self.plot()

#     def plot(
#         self,
#         class_name="attack",
#         alpha_mod=0.5,
#         n_plots=0,
#     ):
#         fig, ax = plt.subplots()
#         ax.plot(
#             self.recall,
#             self.precision,
#             color=f"C{n_plots}",
#         )
#         ax.fill_between(
#             self.recall,
#             self.precision,
#             alpha=0.75 - alpha_mod * n_plots,
#             color=f"C{n_plots}",
#             label=class_name + " (area = {0:0.3f})".format(self.report_dict["pr_auc"]),
#         )
#         ax.fill_between(
#             self.recall,
#             self.precision,
#             alpha=0.75 - alpha_mod * n_plots,
#         )
#         plt.xlim([0.0, 1])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel("TPR")
#         plt.ylabel("PPV")
#         plt.title("Precision-recall curve - binary")
#         plt.legend(loc="upper left")
#         fig.savefig(self.store_path + "binary_pr_curve.png")
#         plt.close(fig)


# class YoudenMetric(ScoreMetric):
#     """Class inherited from ScoreMetric with the goal to calculate the Youden index.

#     Args:
#         data (pd.DataFrame): data used to calculate the metrics
#         true_label (str OR bool): depends on if anomaly or classification model and data
#         score (list): set of score provided by a model or combiner that will be assessed
#         against different metrics.

#     Public Methods:
#         evaluate: calculates the Youden index using the provided list of of values
#         defined by true_label in the dataset against the score.
#     """

#     def __init__(
#         self,
#         data: pd.DataFrame,
#         score: list,
#         true_label: str = TRUE_ANOMALY_LABEL,
#     ):
#         super().__init__(
#             data,
#             true_label,
#         )
#         self.y_score = score

#     def calculate(self):
#         """Generates a dictionary of metrics from the given ROC data based on the Youden Index.

#         Args:
#             data (pd.DataFrame): Dataset
#             fpr (np.ndarray): Array of False Positive Rates
#             tpr (np.ndarray): Array of True Positive Rates
#             thresholds (np.ndarray): Array of thresholds
#         """

#         def get_best_f1(fpr, tpr):
#             """Gets the best F1-score for a given False Positive Rate and True Positive Rate

#             Args:
#                 fpr (np.ndarray): False Positive Rate
#                 tpr (np.ndarray): True Positive Rate

#             Returns:
#                 float: Best F1-score
#             """
#             f1 = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))
#             return f1[np.argmax(f1)]

#         # We need to compute the number of NaN values at the beginning of the score
#         # list so we don't take them into account when calculating the metrics.
#         nans = sum(np.isnan(x) for x in self.y_score)
#         metric_score = self.y_score[nans:]
#         y_true = self.data[self.true_label][nans:]
#         fpr, tpr, thresholds = roc_curve(y_true, metric_score, drop_intermediate=False)

#         max_jindex = np.argmax(tpr - fpr)
#         y_pred = metric_score >= thresholds[max_jindex]
#         # NOTE: zero_division parameter set to 0 applies 0/0 = 0
#         self.report_dict = {
#             "best_f1": get_best_f1(fpr, tpr),
#             "youden_threshold": thresholds[max_jindex],
#             "youden_fpr": fpr[max_jindex],
#             "youden_tpr": tpr[max_jindex],
#             "youden_accuracy": accuracy_score(y_true, y_pred),
#             "youden_precision": precision_score(y_true, y_pred, zero_division=0),
#             "youden_f1": f1_score(y_true, y_pred, zero_division=0),
#         }


class ScorePlotMetric(ScoreMetric):
    """Metric that plots scores over time with background highlighting for anomalies.

    This metric creates a plot where:
    - The x-axis is the instance index
    - The y-axis is the score value
    - The background is colored to indicate ground truth:
        - No color for normal instances
    - Solid color for anomalous instances
    """

    def __init__(
        self,
        store_path: Path,
        y_true: pd.Series,
        y_score: pd.DataFrame,
        level: int = 0,
        y_scale: str = "linear",
        symlog_linthresh: float = 1e-3,
    ) -> None:
        super().__init__(
            store_path=store_path, y_true=y_true, y_score=y_score, level=level
        )
        self.y_scale = y_scale
        self.symlog_linthresh = symlog_linthresh

    def calculate(self) -> pd.DataFrame:
        """Calculates and saves the score plot.

        Returns:
            dict: Empty dict as this metric only produces a plot
        """
        # Get non-NaN indices
        non_nan_mask = ~self.y_score.iloc[:, 0].isna().values.flatten()
        scores = self.y_score[non_nan_mask].iloc[:, 0]  # Take first score column
        true_labels = self.y_true[non_nan_mask]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        if self.y_scale != "linear":
            y_scale = self.y_scale
            if y_scale == "log" and (scores.to_numpy() <= 0).any():
                Module.log_note(
                    self.level,
                    "score_plot y_scale='log' requested but non-positive scores detected; using y_scale='symlog' instead",
                    category="WARNING",
                )
                y_scale = "symlog"

            if y_scale == "symlog":
                ax.set_yscale("symlog", linthresh=self.symlog_linthresh)
            else:
                ax.set_yscale(y_scale)

        # Plot scores
        ax.plot(scores.index.values, scores, "b-", label="Anomaly Score", alpha=0.7)

        # Add background highlighting for anomalies
        anomaly_indices = true_labels[true_labels == True].index
        if len(anomaly_indices) > 0:
            # Get the y-axis limits
            # ymin, ymax = ax.get_ylim()
            # Add background highlighting
            for i, idx in enumerate(anomaly_indices):
                ax.axvspan(
                    idx,
                    idx + 1,
                    ymin=0,
                    ymax=1,
                    color="red",
                    alpha=0.2,
                    label="Anomaly" if i == 0 else None,  # Only label first
                )

        # Customize plot
        ax.set_xlabel("Instance Index")
        ax.set_ylabel("Score")
        ax.set_title("Score Over Time with Anomaly Highlighting")
        ax.grid(True, alpha=0.3)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        # Save plot
        plt.tight_layout()
        plt.savefig(self.store_path / "score_plot.png")
        plt.close()

        return pd.DataFrame()
