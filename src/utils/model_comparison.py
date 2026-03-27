from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aeon.visualisation import plot_critical_difference
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare

FORMAT_DICT = {
    "dataset_name": "Dataset",
    "model_name": "Model",
}


# region Model Acronym Parameter and Structure Extraction Functions
def get_model_parameter(model_name):
    # Extract the first part before the hyphen in the model name
    return model_name.split("-")[0] if "-" in model_name else model_name


def get_model_structure(model_name):
    # Extract the part after the first hyphen in the model name, or empty string if no hyphen
    return model_name.split("-", 1)[1] if "-" in model_name else ""


# endregion


def read_and_combine_experiment_results(
    experiment_name: str,
    dataset_name_list: list[str],
    pipeline_path: Path,
) -> pd.DataFrame:
    """
    Read and combine experiment results from multiple datasets into a single DataFrame.
    This function aggregates CSV results from multiple datasets organized under a specified
    experiment name within a pipeline directory structure. It formats dataset names for
    presentation and creates additional computed columns.
    Parameters
    ----------
    experiment_name : str
        The name of the experiment directory to search for (prefixed with "gs_").
    dataset_name_list : list[str]
        A list of dataset names to process. Each dataset should have a corresponding
        subdirectory in the pipeline path.
    pipeline_path : Path
        The root path where experiment results are organized by dataset name.
        Expected structure: pipeline_path/dataset_name/gs_{experiment_name}/*.csv
    Returns
    -------
    pd.DataFrame
        A combined DataFrame containing all results from the processed datasets with:
        - All original columns from individual CSV files
        - "dataset_name" column: Formatted dataset name with proper capitalization
        - "model_name" column: Concatenation of "parametric" and "structure" columns
    Notes
    -----
    - Skips datasets with no CSV files (prints warning)
    - Uses the lexicographically largest CSV filename when multiple CSV files exist
    - Handles dataset name formatting by preserving uppercase acronyms and capitalizing
      other parts, with special handling for possessive forms ('S -> 's)
    Warnings
    --------
    - Prints a warning if no CSV files are found for a dataset
    - Prints a message if multiple CSV files exist for a dataset
    """

    all_data_list = []
    for dataset_name in dataset_name_list:
        experiment_result_path = pipeline_path / dataset_name / f"gs_{experiment_name}"
        csv_files = list(experiment_result_path.glob("*.csv"))
        if not csv_files:
            print(
                f"Warning: No CSV files found in {experiment_result_path}, skipping this dataset."
            )
            continue
        elif len(csv_files) > 1:
            print(f"dataset {dataset_name} has multiple csv files")
            # NOTE: we keep all csv files for now, as they may contain different results (e.g., from different runs or configurations)
            # removable_csv = max(csv_files, key=lambda f: f.name)
            # removable_csv.unlink()
            # print(f"Removed {removable_csv} to ensure only one CSV file per dataset.")

        # Latest in lexicographical order
        # NOTE: should be max in general
        latest_csv = max(csv_files, key=lambda f: f.name)
        dataset_result_df = pd.read_csv(latest_csv)

        dataset_name_parts = dataset_name.split("_")
        formatted_parts = []
        for part in dataset_name_parts:
            # Preserve acronyms that are already in uppercase
            if part.isupper():
                formatted_parts.append(part)
            # Capitalize only the first letter of other parts
            else:
                formatted_parts.append(part.capitalize())
        dataset_name = " ".join(formatted_parts).replace("'S", "'s")
        # Add new columns efficiently and defragment the DataFrame
        extra_cols = pd.DataFrame(
            {
                "dataset_name": [dataset_name] * len(dataset_result_df),
                "model_name": dataset_result_df["parametric"]
                + dataset_result_df["structure"],
            },
            index=dataset_result_df.index,
        )
        dataset_result_df = pd.concat([dataset_result_df, extra_cols], axis=1)
        all_data_list.append(dataset_result_df)
    experiment_result_df = pd.concat(all_data_list, ignore_index=True)
    return experiment_result_df


def get_metric_matrix(
    experiment_result_df: pd.DataFrame,
    model_name_dict: dict[str, str],
    metric_name: str,
) -> pd.DataFrame:
    """
    Generates a metric matrix DataFrame for specified models and a given metric.
    This function pivots the input experiment results DataFrame to create a matrix where each row corresponds to a dataset,
    each column corresponds to a model from the provided list, and each cell contains the value of the specified metric.
    Args:
        experiment_result_df (pd.DataFrame): DataFrame containing experiment results with columns for 'dataset_name', 'model_name',
            and metric values under the path metric_name.
        model_name_dict (dict[str, str]): Dictionary mapping model names to their short names.
        metric_name (str): Name of the metric to extract from the experiment results.
    Returns:
        pd.DataFrame: A DataFrame where rows are dataset names, columns are model names, and values are the specified metric.
    """

    metric_matrix_df = experiment_result_df.pivot(
        index="dataset_name",
        columns="model_name",
        values=metric_name,
    )
    metric_matrix_df = metric_matrix_df.reindex(columns=list(model_name_dict.keys()))
    metric_matrix_df.rename(columns=model_name_dict, inplace=True)
    return metric_matrix_df


def get_avg_std_metric_matrix(
    experiment_result_df: pd.DataFrame,
    model_name_dict: dict[str, str],
    metric_name: str,
) -> pd.DataFrame:
    """
    Generates a matrix of formatted average $\\pm$ standard deviation metric values for given models and a specified metric.
    Args:
        experiment_result_df (pd.DataFrame): DataFrame containing experiment results, including average and standard deviation metrics for each model and dataset.
        model_name_dict (dict[str, str]): Dictionary mapping model names to their short names.
        metric_name (str): The name of the average metric column (e.g., 'avg_metrics/accuracy').
    Returns:
        pd.DataFrame: A pivoted DataFrame where rows are dataset names, columns are model names (as specified in model_name_dict),
        and values are strings formatted as "avg $\\pm$ std" for the specified metric.
    Notes:
        - Assumes that for each 'avg_metrics/...' column, there is a corresponding 'std_metrics/...' column in the DataFrame.
        - The function formats each metric value to two decimal places.
    """

    avg_metric_name = metric_name
    std_metric_name = metric_name.replace("avg_metrics", "std_metrics")

    simple_metric_name = metric_name.split("/")[-1]
    avg_std_metric_name = f"avg_std_{simple_metric_name}"

    # Use str.cat to join formatted avg and std parts; avoids '+' type error with Series and Literal
    avg_formatted = experiment_result_df[avg_metric_name].map(lambda x: f"{x:.2f}")
    std_formatted = experiment_result_df[std_metric_name].map(lambda x: f"{x:.2f}")
    experiment_result_df[avg_std_metric_name] = avg_formatted.str.cat(
        std_formatted, sep=" $\\pm$ "
    )

    avg_std_metric_matrix = experiment_result_df.pivot(
        index="dataset_name",
        columns="model_name",
        values=avg_std_metric_name,
    )
    avg_std_metric_matrix = avg_std_metric_matrix.reindex(
        columns=list(model_name_dict.keys())
    )
    avg_std_metric_matrix.rename(columns=model_name_dict, inplace=True)
    avg_std_metric_matrix.index.name = FORMAT_DICT["dataset_name"]
    avg_std_metric_matrix.columns.name = FORMAT_DICT["model_name"]

    return avg_std_metric_matrix


def get_ranking_matrix(
    metric_matrix_df: pd.DataFrame,
    model_name_dict: dict[str, str],
    file_name: Path = Path(),
    lower_better: bool = False,
) -> pd.DataFrame:
    """
    Generates a ranking matrix for model comparison, formats the results, and saves the table as an image.
    Args:
        metric_matrix_df (pd.DataFrame): DataFrame containing metric values for each model and dataset.
        model_name_dict (dict[str, str]): Dictionary mapping model names to their short names.
        file_name (Path, optional): Path to save the resulting table image. Defaults to an empty Path.
    Returns:
        pd.DataFrame: DataFrame containing the ranking of each model for each dataset.
    Side Effects:
        - Saves a formatted table as a PNG image at the specified file path.
        - Prints the file path where the table image is saved.
    Notes:
        - The table includes the metric values and their corresponding ranks for each model.
        - Additional rows for the sum and average of ranks are appended to the table.
        - Model names are shortened using the `bn_to_acronym` function.
    """
    formatted_results = metric_matrix_df[model_name_dict.values()].copy()
    ranking_matrix_df = formatted_results.rank(
        axis=1,
        method="min",
        ascending=lower_better,  # If lower_better is True, ascending=True (lower is better)
    )
    for col in formatted_results.columns:
        formatted_results[col] = (
            formatted_results[col].round(3).astype(str)
            + " ("
            + ranking_matrix_df[col].astype(int).astype(str)
            + ")"
        )

    # Add a row for the sum of ranks and average of ranks
    sum_ranks = ranking_matrix_df.sum().round(3).rename("rank_sum")
    average_ranks = ranking_matrix_df.mean().round(3).rename("rank_avg")

    # Add the rows to the formatted DataFrame using concat
    formatted_results = pd.concat(
        [formatted_results, sum_ranks.to_frame().T, average_ranks.to_frame().T]
    )

    # Add the 'Dataset' column to the formatted DataFrame
    formatted_results.insert(
        0,
        FORMAT_DICT["dataset_name"],
        list(metric_matrix_df.index) + ["rank_sum", "rank_avg"],
    )
    # Save the formatted results to a CSV file
    if file_name:
        formatted_results.to_csv(file_name, index=False)

    return ranking_matrix_df


def compute_friedman_test(ranking_matrix_df: pd.DataFrame) -> tuple:
    """
    Compute the Friedman test on the rankings matrix.
    Args:
        df (pd.DataFrame): DataFrame containing model performance metrics.
    Returns:
        tuple: Friedman test statistic and p-value.
    """

    friedman_stat, p_value = friedmanchisquare(*ranking_matrix_df.T.values)
    return friedman_stat, p_value


def plot_critical_difference_diagram(
    metric_matrix_df: pd.DataFrame,
    model_name_dict: dict[str, str],
    file_name: Path = Path(),
    lower_better: bool = False,
    friedman_stat: float | None = None,
    p_value: float | None = None,
) -> np.ndarray | None:
    """
    Plot the critical difference diagram for the rankings matrix.

    Args:
        metric_matrix_df (pd.DataFrame): DataFrame containing the rankings matrix.
        model_name_dict (dict[str, str]): Dictionary mapping model names to their short names.
        file_name (Path): Path to save the critical difference diagram.
        friedman_stat (float, optional): Friedman test statistic to display.
        p_value (float, optional): p-value from the Friedman test to display.
    Returns:
        None: The function saves the critical difference diagram as a PNG file.
    """

    # Generate the critical difference diagram on the provided axis

    label_color_dict = {
        label: (
            "#FFB300"  # Bright orange for SemiParametric (solution, stands out)
            if str(label).startswith("SP")
            else (
                "#1976D2"  # Blue for KDE (alternative)
                if str(label).startswith("KDE")
                else (
                    "#388E3C"  # Green for Gaussian (alternative)
                    if str(label).startswith("G")
                    else "#757575"  # Gray for others (alternative)
                )
            )
        )
        for label in model_name_dict.values()
    }
    scores = metric_matrix_df[list(model_name_dict.values())].values
    result = plot_critical_difference(
        scores=scores,
        labels=list(model_name_dict.values()),
        highlight=label_color_dict,
        lower_better=lower_better,  # higher metric is better
        test="wilcoxon",  # or nemenyi
        correction="holm",  # or bonferroni or none
        alpha=0.05,  # significance level
        reverse=False,
        return_p_values=True,
    )
    if len(result) == 3:
        fig, ax, p_values = result
    else:
        fig, ax = result
        p_values = None

    # Adjust font size and rotation of x-axis labels
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    # Increase padding between labels and axis
    ax.tick_params(axis="x", which="major", pad=20)

    # Adjust margins to provide more space for labels
    fig.subplots_adjust(bottom=0.35)

    # Optionally adjust y-axis label font size
    ax.tick_params(axis="y", labelsize=12)

    # Add Friedman test statistic and p-value to the plot
    if friedman_stat is not None and p_value is not None:
        text = f"Friedman χ² = {friedman_stat:.3f}\np-value = {p_value:.3g}"
        # Place the text in the top-left corner with more margin
        ax.text(
            -0.15,  # Move even further left to avoid overlap
            1.05,  # Move higher above the plot
            text,
            transform=ax.transAxes,
            fontsize=12,  # Slightly smaller font
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", pad=5),
            clip_on=False,
        )

        # Adjust the figure layout to accommodate the text
        fig.subplots_adjust(bottom=0.35, top=0.85, left=0.15)

    # Save and display the plot
    fig.savefig(
        file_name,
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    return p_values


def plot_sp_structure_boxplot(
    metric_matrix_df: pd.DataFrame, simple_metric_name: str, output_dir: Path
) -> None:
    """
    Generate and save a boxplot comparing metric values across different structures
    for semi-parametric (SP) models.
    This function creates a publication-quality boxplot visualizing the distribution
    of a specified metric across multiple model structures, filtering for only
    semi-parametric models. The plot includes median lines, mean lines, and styled
    outliers with a colorblind-friendly color scheme.
    Args:
        metric_matrix_df (pd.DataFrame): A DataFrame where column names follow the
            naming convention "{Parametric}-{Structure}" (e.g., "SP-BN", "SP-TAN").
            Column values should contain numeric metric values for each observation.
        simple_metric_name (str): The name of the metric being visualized, used for
            plot labels and output filename. Will be beautified by replacing underscores
            with spaces. Special cases include "roc_auc" → "ROC AUC" and
            "log_likelihood" → "Log-Likelihood".
        output_dir (Path): The directory path where the output boxplot PNG image
            will be saved.
    Returns:
        None: Saves the generated boxplot as a PNG file to the specified output
            directory with the filename format "{metric_name}_sp_structure_boxplot.png".
    Notes:
        - Only filters and plots columns where the parametric type is "SP"
        - Uses a colorblind-friendly palette with up to 7 distinct colors
        - Median is shown as a solid black line; mean is shown as a dashed red line
        - Outliers are displayed as red circles
        - Saved image has 300 DPI for publication-quality output
        - If no SP models are found in the data, no plot is generated
    """
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
        legend_elements = [
            Line2D([0], [0], color="black", linewidth=2, label="Median"),
            Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="Mean"),
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
            output_dir / f"{simple_metric_name}_sp_structure_boxplot.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()


def bold_best_cell(row: pd.Series, lower_better: bool = False) -> pd.Series:
    """
    Bold the best performing cell(s) in each row of a pandas Series.
    This function identifies the best cell(s) based on mean values extracted from
    "mean ± std" formatted strings, and applies LaTeX bold formatting to them.
    When multiple cells have the same best mean, the one(s) with the lowest
    standard deviation are bolded.
    Parameters
    ----------
    row : pd.Series
        A pandas Series where each cell contains a string in the format
        "mean ± std" (e.g., "0.95 ± 0.02").
    lower_better : bool, optional
        If True, treats lower mean values as better (e.g., for error metrics).
        If False (default), treats higher mean values as better (e.g., for accuracy).
    Returns
    -------
    pd.Series
        A copy of the input row with the best cell(s) wrapped in LaTeX bold
        formatting using \\textbf{} notation.
    Examples
    --------
    >>> row = pd.Series({'Model_A': '0.95 ± 0.02', 'Model_B': '0.92 ± 0.03'})
    >>> bold_best_cell(row, lower_better=False)
    Model_A    \\textbf{0.95 ± 0.02}
    Model_B                0.92 ± 0.03
    dtype: object
    """

    # Bold the best mean $\\pm$ std cell in each row
    # Extract means and stds from "mean $\\pm$ std" formatted strings
    def parse_mean_std(cell):
        parts = str(cell).split("$\\pm$")
        mean = float(parts[0].strip())
        std = float(parts[1].strip()) if len(parts) > 1 else float("inf")
        return mean, std

    stats = row.apply(parse_mean_std)
    means = stats.apply(lambda x: x[0])
    stds = stats.apply(lambda x: x[1])

    if lower_better:
        best_mean = means.min()
    else:
        best_mean = means.max()

    # Find all indices with the best mean
    best_indices = means[means == best_mean].index
    # Among them, pick all with the lowest std
    best_std = stds[best_indices].min()
    best_std_indices = stds[best_indices][stds[best_indices] == best_std].index

    # Bold all best cells
    new_row = row.copy()
    for idx in best_std_indices:
        new_row[idx] = f"\\textbf{{{row[idx]}}}"
    return new_row
