import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# TODO: Remove dependence
RUTILE_AI_PATH = Path("/app/dev/rutile-ai")
DATA_PATH = RUTILE_AI_PATH / "data"
PIPELINE_PATH = DATA_PATH / "pipelines"
RESULT_PATH = DATA_PATH / "agg_results"
AVG_STD_RESULT_PATH = RESULT_PATH / "avg_std_tables"
METRIC_MATRIX_RESULT_PATH = RESULT_PATH / "metric_matrices"
RANKING_TABLES_RESULT_PATH = RESULT_PATH / "ranking_tables"
CD_DIAGRAMS_RESULT_PATH = RESULT_PATH / "cd_diagrams"

sys.path.append(str(RUTILE_AI_PATH))

from rutile_ai.data_handler import DATASET_NAME_LIST
from rutile_ai.pipeline.evaluator.model_comparison import (
    bold_best_cell,
    get_avg_std_metric_matrix,
    get_metric_matrix,
    get_model_parameter,
    get_model_structure,
    get_ranking_matrix,
    plot_critical_difference_diagram,
    read_and_combine_experiment_results,
)
from src.spbnclassify.bn import bn_to_acronym

EXPERIMENT_NAME = f"bnc"

MODEL_NAME_LIST = [
    # Gaussian classifiers
    "GaussianNaiveBayes",
    "GaussianSelectiveNaiveBayes",
    "GaussianTreeAugmentedNaiveBayes",
    "GaussianAveragedOneDependenceEstimator",
    "GaussianKDependenceBayesian",
    "GaussianBayesianNetworkAugmentedNaiveBayes",
    "GaussianBayesianMultinet",
    # KDE classifiers
    "KDENaiveBayes",
    "KDESelectiveNaiveBayes",
    "KDETreeAugmentedNaiveBayes",
    "KDEAveragedOneDependenceEstimator",
    "KDEKDependenceBayesian",
    "KDEBayesianNetworkAugmentedNaiveBayes",
    "KDEBayesianMultinet",
    # SemiParametric classifiers
    "SemiParametricNaiveBayes",
    "SemiParametricSelectiveNaiveBayes",
    "SemiParametricTreeAugmentedNaiveBayes",
    "SemiParametricAveragedOneDependenceEstimator",
    "SemiParametricKDependenceBayesian",
    "SemiParametricBayesianNetworkAugmentedNaiveBayes",
    "SemiParametricBayesianMultinet",
]

MODEL_NAME_DICT = {name: bn_to_acronym(name) for name in MODEL_NAME_LIST}
METRIC_CONFIG_DICT = {
    "cross_validation_avg/avg_metrics/weighted avg/accuracy": {"lower_better": False},
    "cross_validation_avg/avg_metrics/weighted avg/F1-score": {"lower_better": False},
    "cross_validation_avg/avg_metrics/weighted avg/ROC_AUC": {"lower_better": False},
    "cross_validation_avg/avg_metrics/macro avg/log_likelihood": {
        "lower_better": False
    },
    "cross_validation_avg/avg_metrics/macro avg/training_time": {"lower_better": True},
    "cross_validation_avg/avg_metrics/macro avg/testing_time": {"lower_better": True},
    "cross_validation_avg/avg_metrics/macro avg/parametric_node_type_ratio": {
        "lower_better": False
    },
}
CLASS_METRIC_LIST = ["accuracy", "F1-score", "AUC", "log_likelihood"]

if __name__ == "__main__":
    AVG_STD_RESULT_PATH.mkdir(parents=True, exist_ok=True)
    METRIC_MATRIX_RESULT_PATH.mkdir(parents=True, exist_ok=True)
    RANKING_TABLES_RESULT_PATH.mkdir(parents=True, exist_ok=True)
    CD_DIAGRAMS_RESULT_PATH.mkdir(parents=True, exist_ok=True)

    # region Read and combine experiment results
    experiment_result_df = read_and_combine_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        dataset_name_list=DATASET_NAME_LIST,
        pipeline_path=PIPELINE_PATH,
    )
    experiment_file_name = RESULT_PATH / f"{EXPERIMENT_NAME}_combined_results.csv"
    experiment_result_df.to_csv(experiment_file_name, index=False)
    # endregion Read and combine experiment results

    # region Model Comparison for each metric
    all_rankings_dict = {}
    for metric_name in METRIC_CONFIG_DICT.keys():
        # region Parameter setup
        simple_metric_name = metric_name.split("/")[-1]
        if simple_metric_name == "ROC_AUC":
            simple_metric_name = "AUC"
        lower_better = METRIC_CONFIG_DICT[metric_name]["lower_better"]

        metric_file_name = (
            METRIC_MATRIX_RESULT_PATH / f"{simple_metric_name}_metric_matrix.csv"
        )
        summary_metric_latex_name = (
            METRIC_MATRIX_RESULT_PATH / f"{simple_metric_name}_metric_summary.tex"
        )
        avg_std_csv_name = (
            AVG_STD_RESULT_PATH / f"{simple_metric_name}_avg_std_metric_matrix.csv"
        )
        avg_std_latex_name = (
            AVG_STD_RESULT_PATH / f"{simple_metric_name}_avg_std_metric_matrix.tex"
        )
        ranking_file_name = (
            RANKING_TABLES_RESULT_PATH / f"{simple_metric_name}_ranking_table.csv"
        )
        summary_ranking_latex_name = (
            RANKING_TABLES_RESULT_PATH / f"{simple_metric_name}_ranking_summary.tex"
        )
        cd_file_name = CD_DIAGRAMS_RESULT_PATH / f"{simple_metric_name}_cd_diagram.png"
        # endregion Parameter setup

        # region Metric Matrix
        # Calculate the metric matrix
        metric_matrix_df = get_metric_matrix(
            experiment_result_df, MODEL_NAME_DICT, metric_name
        )
        metric_matrix_df.to_csv(metric_file_name, index=True)

        # Ensure non-NaN values are present
        # Drop rows with NaN values to ensure valid comparisons
        na_rows = metric_matrix_df.isna().any(axis=1).sum()
        na_row_indices = metric_matrix_df.index[
            metric_matrix_df.isna().any(axis=1)
        ].tolist()

        if na_rows > 0:
            print(
                f"Warning: {na_rows} row(s) with NaN values removed from metric matrix for metric: {simple_metric_name} with dataset(s): {na_row_indices}"
            )
        metric_matrix_df = metric_matrix_df.dropna()
        # RFE: Refactor in function
        # Boxplot for SemiParametric models by Structure
        sp_parametric = "SP"
        labels = []
        boxplot_data = []

        # For each Structure, plot a boxplot of the metric values for Parametric == SP
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in metric_matrix_df.columns:
            parametric = col.split("-")[0]
            structure = col.split("-")[1]
            if parametric != sp_parametric:
                continue

            boxplot_data.append(metric_matrix_df[col].values)
            labels.append(col)  # Use just the structure name for cleaner labels

        if boxplot_data:
            # Create boxplot with improved styling
            bp = ax.boxplot(
                boxplot_data,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                # notch=True,
            )

            # Color scheme for academic papers (colorblind-friendly)
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ]

            # Style the boxplot elements
            for i, (patch, median, mean) in enumerate(
                zip(bp["boxes"], bp["medians"], bp["means"])
            ):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
                patch.set_edgecolor("black")
                patch.set_linewidth(1.2)

                # Style median line
                median.set_color("black")
                median.set_linewidth(2)

                # Style mean line
                mean.set_color("red")
                mean.set_linewidth(2)
                mean.set_linestyle("--")

            # Style whiskers and caps
            for whisker in bp["whiskers"]:
                whisker.set_color("black")
                whisker.set_linewidth(1.5)
                whisker.set_linestyle("-")

            for cap in bp["caps"]:
                cap.set_color("black")
                cap.set_linewidth(1.5)

            # Style outliers
            for flier in bp["fliers"]:
                flier.set_marker("o")
                flier.set_markerfacecolor("red")
                flier.set_markeredgecolor("red")
                flier.set_markersize(4)
                flier.set_alpha(0.6)

            # Beautify the y-axis label for better readability
            beautified_label = simple_metric_name.replace("_", " ").title()
            if beautified_label == "Roc Auc":
                beautified_label = "ROC AUC"
            elif beautified_label == "Log Likelihood":
                beautified_label = "Log-Likelihood"

            # Set labels and title
            ax.set_xlabel("Model", fontsize=12, fontweight="bold")
            ax.set_ylabel(beautified_label, fontsize=12, fontweight="bold")

            # Set x-tick labels
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=11, rotation=0)

            # Improve tick formatting
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.tick_params(axis="x", which="major", pad=5)

            # Extend y-axis above 1.0 to accommodate legend
            ax.set_ylim(bottom=-0.05, top=1.1)  # Or use top=1.05 for less space

            # Set background and grid
            ax.set_facecolor("white")
            ax.grid(
                True,
                which="major",
                axis="y",
                linestyle="-",
                linewidth=0.5,
                color="gray",
                alpha=0.3,
            )
            ax.set_axisbelow(True)

            # Add legend for mean and median
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="black", linewidth=2, label="Median"),
                Line2D(
                    [0], [0], color="red", linewidth=2, linestyle="--", label="Mean"
                ),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=10,
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
            )

            # Adjust layout and add border
            plt.tight_layout()
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color("black")

            # Save with high DPI for publication quality
            plt.savefig(
                METRIC_MATRIX_RESULT_PATH
                / f"{simple_metric_name}_sp_structure_boxplot.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close()

        # Aggregate mean and std of each column (model) in the metric matrix
        metric_means = metric_matrix_df.mean(axis=0)
        metric_stds = metric_matrix_df.std(axis=0)
        metric_summary_df = pd.DataFrame(
            {"Model": metric_means.index, "Mean": metric_means, "STD": metric_stds}
        )
        metric_summary_df["Family"] = metric_summary_df["Model"].apply(
            get_model_parameter
        )
        metric_summary_df["Model"] = metric_summary_df["Model"].apply(
            get_model_structure
        )

        # Retain original Structure and Parametric ordering in the pivot
        structure_order = metric_summary_df["Model"].unique()
        parametric_order = metric_summary_df["Family"].unique()

        # Metric Summary
        metric_summary_df = metric_summary_df.pivot(
            index="Model",
            columns="Family",
            values=["Mean", "STD"],
        )
        # Reindex to preserve original order
        metric_summary_df = metric_summary_df.reindex(index=structure_order)
        metric_summary_df = metric_summary_df.reindex(columns=parametric_order, level=1)

        # Combine mean and std into a single string per cell
        metric_summary_df = metric_summary_df.apply(
            lambda row: {
                col: f"{row['Mean'][col]:.2f} $\\pm$ {row['STD'][col]:.2f}"
                for col in row["Mean"].index
            },
            axis=1,
            result_type="expand",
        )
        metric_summary_df.index.name = "Model"
        metric_summary_df.columns.name = "Family"
        latex_df = metric_summary_df.copy()
        latex_df = latex_df.round(2)
        latex_bold_df = latex_df.apply(
            bold_best_cell, lower_better=lower_better, axis=1
        )
        # RFE: Add \resizebox{\textwidth}{!} before \begin{tabular} and closing bracket after \end{tabular}
        latex_bold_df.to_latex(
            buf=summary_metric_latex_name,
            float_format="%.2f",
            caption=f"Average metric for each model over all datasets (mean $\\pm$ standard deviation). The best (highest) results are highlighted in bold. In case of a draw, all best results are highlighted.",
            label=f"tab:{simple_metric_name}_metric_summary",
            # column_format=col_format,
            escape=False,
            multicolumn=True,
            multicolumn_format="c",
            position="htbp",
        )
        # endregion Metric Matrix

        # region Average and Standard Deviation Metric Matrix
        avg_std_metric_matrix = get_avg_std_metric_matrix(
            experiment_result_df, MODEL_NAME_DICT, metric_name
        )

        avg_std_metric_matrix.to_csv(avg_std_csv_name, index=True)

        # Export to LaTeX table
        latex_df = avg_std_metric_matrix.copy()
        latex_df.columns.name = "Model"
        latex_df.index.name = "Dataset"

        latex_bold_df = latex_df.apply(bold_best_cell, axis=1)

        # Add two vertical separators equally distributed in the column format
        num_cols = latex_bold_df.shape[1]
        # For example, for 6 columns: l|cc|cc|c
        # Calculate positions for separators
        sep_positions = [0, num_cols // 3, 2 * num_cols // 3]
        col_format = "l"
        for i in range(num_cols):
            if i in sep_positions:
                col_format += "|"
            col_format += "c"

        def sci_fmt(x):
            # Use scientific notation if abs(x) >= 1e4 or abs(x) < 1e-2 and not zero
            if pd.isna(x):
                return ""
            if abs(x) >= 1e4 or (abs(x) < 1e-2 and x != 0):
                return f"{x:.2e}"
            else:
                return f"{x:.2f}"

        # Format both mean and std in each cell using sci_fmt
        def format_mean_std_cell(cell):
            try:
                mean_str, std_str = cell.split(" $\\pm$ ")
                mean_val = float(mean_str)
                std_val = float(std_str)
                return f"{sci_fmt(mean_val)} $\\pm$ {sci_fmt(std_val)}"
            except Exception:
                return cell

        latex_bold_df = latex_bold_df.applymap(format_mean_std_cell)

        latex_bold_df.to_latex(
            buf=avg_std_latex_name,
            float_format=sci_fmt,
            caption=f"Mean and standard deviation of {simple_metric_name.replace('_', ' ')} for each model and dataset (mean $\\pm$ standard deviation). The best (highest) results are highlighted in bold. In case of a draw, all best results are highlighted.",
            label=f"tab:{simple_metric_name}_avg_std",
            column_format=col_format,
            escape=False,
            multicolumn=True,
            multicolumn_format="c",
            position="htbp",
        )
        # endregion Average and Standard Deviation Metric Matrix

        # region Critical Difference Diagram
        # Compute the Friedman test
        # friedman_stat, p_value = compute_friedman_test(ranking_matrix_df)

        p_values = plot_critical_difference_diagram(
            metric_matrix_df,
            MODEL_NAME_DICT,
            file_name=cd_file_name,
            lower_better=lower_better,
            # friedman_stat=friedman_stat,
            # p_value=p_value,
        )
        # endregion Critical Difference Diagram

        # region Ranking Table
        ranking_matrix_df = get_ranking_matrix(
            metric_matrix_df,
            MODEL_NAME_DICT,
            file_name=ranking_file_name,
            lower_better=lower_better,
        )
        # Calculate average ranking for each model across all metrics and datasets
        average_rankings_mean = ranking_matrix_df.mean()
        # Calculate standard deviation for each model across all metrics and datasets
        average_rankings_std = ranking_matrix_df.std()

        # Create a summary DataFrame
        ranking_summary_df = pd.DataFrame(
            {
                "Model": average_rankings_mean.index,
                "Mean Ranking": average_rankings_mean.values,
                "STD Ranking": average_rankings_std.values,
            }
        )

        ranking_summary_df["Family"] = ranking_summary_df["Model"].apply(
            get_model_parameter
        )
        ranking_summary_df["Model"] = ranking_summary_df["Model"].apply(
            get_model_structure
        )
        # Retain original Structure and Parametric ordering in the pivot
        structure_order = ranking_summary_df["Model"].unique()
        parametric_order = ranking_summary_df["Family"].unique()
        ranking_summary_df = ranking_summary_df.pivot(
            index="Model",
            columns="Family",
            values=["Mean Ranking", "STD Ranking"],
        )
        # Reindex to preserve original order
        ranking_summary_df = ranking_summary_df.reindex(index=structure_order)
        ranking_summary_df = ranking_summary_df.reindex(
            columns=parametric_order, level=1
        )
        # Combine mean and std into a single string per cell
        ranking_summary_df = ranking_summary_df.apply(
            lambda row: {
                col: f"{row['Mean Ranking'][col]:.2f} $\\pm$ {row['STD Ranking'][col]:.2f}"
                for col in row["Mean Ranking"].index
            },
            axis=1,
            result_type="expand",
        )
        ranking_summary_df.index.name = "Model"
        ranking_summary_df.columns.name = "Family"
        latex_df = ranking_summary_df.copy()
        latex_df = latex_df.round(2)
        latex_bold_df = latex_df.apply(bold_best_cell, lower_better=True, axis=1)
        latex_bold_df.to_latex(
            buf=summary_ranking_latex_name,
            float_format="%.2f",
            caption=f"Average ranking for each model over all datasets (mean $\\pm$ standard deviation).",
            label=f"tab:{simple_metric_name}_ranking_summary",
            # column_format=col_format,
            escape=False,
            multicolumn=True,
            multicolumn_format="c",
            position="htbp",
        )
        ranking_matrix_df["Metric"] = simple_metric_name.replace("_", " ").title()
        all_rankings_dict[simple_metric_name] = ranking_matrix_df
        # endregion Ranking Table

    print(f"Results saved in {RESULT_PATH}")
    # endregion

    # region Concat model rankings across all classification metrics
    combined_rankings_df = pd.concat(
        [df for key, df in all_rankings_dict.items() if key in CLASS_METRIC_LIST],
        ignore_index=True,
    )

    average_rankings_mean = combined_rankings_df.groupby("Metric").mean()
    average_rankings_std = combined_rankings_df.groupby("Metric").std()
    # Create a DataFrame with "mean $\\pm$ std" for each cell
    combined_ranking_summary_df = pd.DataFrame(
        {
            model: [
                f"{average_rankings_mean.loc[metric, model]:.2f} $\\pm$ {average_rankings_std.loc[metric, model]:.2f}"
                for metric in average_rankings_mean.index
            ]
            for model in average_rankings_mean.columns
        },
        index=average_rankings_mean.index,
    )
    combined_ranking_summary_df = combined_ranking_summary_df.T
    combined_ranking_summary_df["Family"] = combined_ranking_summary_df.index.map(
        get_model_parameter
    )
    combined_ranking_summary_df["Model"] = combined_ranking_summary_df.index.map(
        get_model_structure
    )
    # Retain original Structure and Parametric ordering in the pivot
    structure_order = combined_ranking_summary_df["Model"].unique()
    parametric_order = combined_ranking_summary_df["Family"].unique()

    combined_ranking_summary_df = combined_ranking_summary_df.pivot(
        index="Model",
        columns="Family",
    )
    # Reindex to preserve original order
    combined_ranking_summary_df = combined_ranking_summary_df.reindex(
        index=structure_order
    )
    combined_ranking_summary_df = combined_ranking_summary_df.reindex(
        columns=parametric_order, level=1
    )
    # latex export
    latex_df = combined_ranking_summary_df.copy()
    latex_df = latex_df.round(2)
    ranking_summary_latex_name = RESULT_PATH / f"agg_ranking_summary.tex"
    # Apply bold_best_cell per Metric column (i.e., per metric for each structure)
    latex_bold_df = latex_df.copy()
    for metric in latex_df.columns.levels[0]:
        latex_bold_df[metric] = latex_df[metric].apply(
            bold_best_cell, lower_better=True, axis=1
        )
    # Split metrics into two groups (2 metrics per table)
    metric_groups = [
        latex_bold_df.columns.levels[0][i : i + 2]
        for i in range(0, len(latex_bold_df.columns.levels[0]), 2)
    ]
    for idx, metrics in enumerate(metric_groups):
        sub_df = latex_bold_df[metrics]
        # Dynamically build column_format with vertical separators for each Metric
        num_metrics = len(sub_df.columns.levels[0])
        num_parametric = len(sub_df.columns.levels[1])
        col_format = "l"
        for i in range(len(metrics)):
            col_format += "|"
            col_format += "c" * num_parametric

        sub_latex_name = RESULT_PATH / f"agg_ranking_summary_part{idx+1}.tex"
        sub_df.to_latex(
            buf=sub_latex_name,
            float_format="%.2f",
            caption=f"Mean and standard deviation of average rankings for each model and family across all datasets, evaluated for each performance metric: {', '.join(metrics)}. For each structure (in rows), the best (lowest) configuration is highlighted in bold.",
            label=f"tab:agg_ranking_summary_part{idx+1}",
            column_format=col_format,
            escape=False,
            multicolumn=True,
            multicolumn_format="c",
            position="htbp",
        )
