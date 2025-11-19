from pathlib import Path

import numpy as np
import pandas as pd
import pyagrum as gum
import pyagrum.causal as csl
import pyagrum.lib.explain as expl
import pyagrum.lib.image as gumimage
import pyagrum.lib.notebook as gnb
import pyagrum.lib.shapley as shapley
import pybnesian as pbn
import pydot
from matplotlib import pyplot as plt
from rutile_ai.data_handler import TRUE_ANOMALY_LABEL

from .base import BayesianNetwork
from .utils import convert_arcs_to_names


class DiscreteBayesianNetwork(
    BayesianNetwork, pbn.DiscreteBN
):  # Method Resolution Order important (save/load)
    bn_type = pbn.DiscreteBNType()
    search_operators = ["arcs"]

    def __init__(
        self,
        search_score: str = "bic",
        arc_blacklist: list[tuple[str, str]] = [],
        arc_whitelist: list[tuple[str, str]] = [],
        type_blacklist: list[tuple[str, pbn.FactorType]] = [],
        type_whitelist: list[tuple[str, pbn.FactorType]] = [],
        callback: pbn.Callback = None,
        max_indegree: int = 0,
        max_iters: int = 2147483647,
        epsilon: int = 0,
        patience: int = 0,
        seed: int | None = None,
        num_folds: int = 5,
        test_holdout_ratio: float = 0.2,
        max_train_data_size: int = 0,
        verbose: bool = False,
        feature_names_in_: list[str] = [],
        n_features_in_: int = 0,
        true_label: str = TRUE_ANOMALY_LABEL,
        prediction_label: str = "binary_predicted_label",
        arc_matrix_df: pd.DataFrame = pd.DataFrame(),
    ) -> None:
        """Initializes the Discrete Bayesian Network with the nodes, arcs, node_types and the structure learning parameters

        Args:
            nodes (list[str], optional): list of nodes. Defaults to [].
            arcs (list[tuple[str, str]], optional): list of arcs. Defaults to [].
            node_types (list[tuple[str, pbn.FactorType]], optional): list of node types. Defaults to [].
            search_score (str, optional): Search score to be used for the structure learning. The possible scores ((validate_options.cpp)) are:
                - "bic" (Bayesian Information Criterion)
                - "bge" (Bayesian Gaussian equivalent)
                - "cv-lik" (Cross-Validated likelihood)
                - "holdout-lik" (Hold-out likelihood)
                - "validated-lik" (Validated likelihood with cross-validation). Defaults to "bic".
            arc_blacklist (list[tuple[str, str]], optional): Arc blacklist (forbidden arcs). Defaults to [].
            arc_whitelist (list[tuple[str, str]], optional): Arc whitelist (forced arcs). Defaults to [].
            type_blacklist (list[tuple[str, pbn.FactorType]], optional): Node type blacklist (forbidden node types). Defaults to [].
            type_whitelist (list[tuple[str, pbn.FactorType]], optional): Node type whitelist (forced node types). Defaults to [].
            callback (pbn.Callback, optional): Callback function to be called after each iteration. Defaults to None.
            max_indegree (int, optional): Maximum indegree allowed in the graph. Defaults to 0.
            max_iters (int, optional): Maximum number of search iterations. Defaults to 2147483647.
            epsilon (int, optional): Minimum delta score allowed for each operator. If the new operator is less than epsilon, the search process is stopped. Defaults to 0.
            patience (int, optional): The patience parameter (only used with pbn.ValidatedScore). Defaults to 0.
            seed (int | None, optional): Seed parameter of the score (if needed). Defaults to None.
            num_folds (int, optional): Number of folds for the CVLikelihood and ValidatedLikelihood scores. Defaults to 5.
            test_holdout_ratio (float, optional): Parameter for the HoldoutLikelihood and ValidatedLikelihood scores. Defaults to 0.2.
            max_train_data_size (int, optional): Maximum sample size to be used for the structure learning. Defaults to 0.
            verbose (bool, optional): If True the progress will be displayed, otherwise nothing will be displayed. Defaults to False.
            true_label (str, optional): The true label column name. Defaults to TRUE_ANOMALY_LABEL.
            prediction_label (str, optional): The predicted label column name. Defaults to "binary_predicted_label".
        """
        pbn.DiscreteBN.__init__(self, nodes=[])
        for source, row in arc_matrix_df.iterrows():
            for target, value in row.items():
                if source != target:
                    if value == 0:  # Forbidden arc
                        arc_blacklist.append((str(source), str(target)))
                    elif value == 1:  # Mandatory arc
                        arc_whitelist.append((str(source), str(target)))
        BayesianNetwork.__init__(
            self,
            search_score=search_score,
            arc_blacklist=arc_blacklist,
            arc_whitelist=arc_whitelist,
            type_blacklist=type_blacklist,
            type_whitelist=type_whitelist,
            callback=callback,
            max_indegree=max_indegree,
            max_iters=max_iters,
            epsilon=epsilon,
            patience=patience,
            seed=seed,
            num_folds=num_folds,
            test_holdout_ratio=test_holdout_ratio,
            max_train_data_size=max_train_data_size,
            verbose=verbose,
            feature_names_in_=feature_names_in_,
            n_features_in_=n_features_in_,
            true_label=true_label,
            prediction_label=prediction_label,
        )

    def __str__(self) -> str:
        """Returns the string representation of the Discrete Bayesian Network

        Returns:
            str: The string representation
        """
        return "Discrete " + BayesianNetwork.__str__(self)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "BayesianNetwork":
        """Learns the structure and parameters of the Discrete Bayesian Network from the data.
        NOTE: pbn.DiscreteBN learns without smoothing and can only be set by fitting data. CPTs might have a slight difference with pyagrum
        Args:
            data (pd.DataFrame): The data to learn from.
        """
        BayesianNetwork.fit(self, X)
        pbn.DiscreteBN.fit(self, X.astype(str).astype("category"))

        return self

    def infer(
        self,
        evidence: dict[str, float] = {},
        targets: set[str] = set(),
        model_file: Path | None = None,
    ) -> str:
        """
        Performs inference on the Bayesian network using the provided evidence and target nodes.
        Args:
            evidence (dict[str, pydot.Dot], optional): A dictionary mapping node names to their observed values. Defaults to an empty dictionary. We can have hard evidence (e.g., {"Execution": True}) or soft evidence (e.g., {"Execution": [0.3, 0.9]}).
            targets (set[str], optional): A set of node names for which to compute the inference results. Defaults to an empty set.
            model_file (Path | None, optional): If provided, exports the HTML inference results to this file.
        Returns:
            str: An HTML string representing the inference results.
        """
        html_str = gnb.getInference(
            self.graphic,
            engine=None,
            evs=evidence,
            targets=targets,
            size="12",
        )

        if model_file:
            gumimage.exportInference(self.graphic, str(model_file.with_suffix(".pdf")))
        return html_str

    def conditional_shap(
        self,
        data: pd.DataFrame,
        target: str,
        plot_file_path: Path | None = None,
        graph_file_path: Path | None = None,
    ) -> tuple[dict, pydot.Dot]:
        """
        Computes conditional SHAP (SHapley Additive exPlanations) values for a given target variable using a graphical model,
        generates visualizations of SHAP values and feature importances, and optionally saves the plots and a SHAP value graph to files.
        This method calculates SHAP values for the Markov Blanket of the target variable, visualizes the results using customized
        beeswarm and feature importance plots, and returns both the mean absolute SHAP values per feature and a graph object
        representing the SHAP value structure.
        Parameters
        ----------
        data : pd.DataFrame
            Input data for which to compute conditional SHAP values.
        target : str
            The name of the target variable for SHAP analysis.
        plot_file_path : Path or None, optional
            File path to save the SHAP value plot. If None, the plot is displayed instead of being saved.
        graph_file_path : Path or None, optional
            File path to save the SHAP value graph as a PDF.
        Returns
        -------
        tuple[dict, pydot.Dot]
            - node_shapvalue_dict : dict
                Dictionary mapping each feature (column name) to its mean absolute conditional SHAP value.
            - graph : pydot.Dot
                A graph object representing the SHAP value structure, saved as a PDF if `graph_file_path` is provided.
        Notes
        -----
        - The target variable is a black node.
        - The most important variable to explain the prediction of the target is a pink node.
        - The least important variable to explain the prediction of the target is a blue node.
        """

        def _plotResults(
            gumshap: expl.ShapValues,
            results: pd.DataFrame,
            v: pd.DataFrame,
            plot: bool = True,
            plot_importance: bool = True,
            percentage: bool = False,
        ):
            """
            Overrides the default expl.ShapValues._plotResults method to customize the visualization of SHAP values.
            Plots SHAP analysis results, including a beeswarm plot of SHAP values and/or a feature importance bar plot.

            Parameters:
                gumshap: object
                    An object with plotting methods `_plot_beeswarm_` and `_plot_importance`.
                results: pandas.DataFrame
                    DataFrame containing SHAP values for each feature (columns) and sample (rows).
                v: pandas.DataFrame
                    DataFrame of feature values, aligned with `results` columns.
                plot: bool, optional (default=True)
                    Whether to display the SHAP values distribution (beeswarm) plot.
                plot_importance: bool, optional (default=True)
                    Whether to display the feature importance bar plot.
                percentage: bool, optional (default=False)
                    If True, show feature importance as percentages; otherwise, show raw values.

            Returns:
                None

            Notes:
                - At least one of `plot` or `plot_importance` must be True to generate a plot.
                - The function customizes figure size and layout based on the number of features and selected plots.
                - Plots are rendered using matplotlib.
            """
            ax1 = ax2 = None
            if plot and plot_importance:
                # Improved figure sizing and layout
                fig = plt.figure(figsize=(16, max(8, 0.6 * len(results.columns))))
                fig.suptitle("SHAP Analysis Results", fontsize=16, fontweight="bold")
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                # Add spacing between subplots
                plt.subplots_adjust(wspace=0.5, top=0.9)
            elif plot:
                fig = plt.figure(figsize=(10, max(6, 0.6 * len(results.columns))))
                ax1 = fig.add_subplot(1, 1, 1)
            elif plot_importance:
                fig = plt.figure(figsize=(8, 6))
                ax2 = fig.add_subplot(1, 1, 1)

            if plot:
                shap_dict = results.to_dict(orient="list")
                sorted_dict = dict(
                    sorted(
                        shap_dict.items(),
                        key=lambda x: sum(abs(i) for i in x[1]) / len(x[1]),
                    )
                )
                data = np.array([sorted_dict[key] for key in sorted_dict])
                features = list(sorted_dict.keys())
                v = v[features]
                colors = v.transpose().to_numpy()

                # Improved beeswarm plot with better styling
                gumshap._plot_beeswarm_(data, colors, 250, 1.5, features, ax=ax1)
                if ax1:
                    ax1.set_title(
                        "SHAP Values Distribution",
                        fontsize=14,
                        fontweight="bold",
                        pad=20,
                    )
                    ax1.grid(True, alpha=0.3)
                    ax1.set_xlabel("SHAP Value", fontsize=12)
                    # Improve y-axis labels
                    ax1.tick_params(axis="y", labelsize=10)
                    ax1.tick_params(axis="x", labelsize=10)

            if plot_importance:
                gumshap._plot_importance(results, percentage, ax=ax2)
                if ax2:
                    ax2.set_title(
                        "Feature Importance", fontsize=14, fontweight="bold", pad=20
                    )
                    ax2.grid(True, alpha=0.3)
                    ax2.tick_params(axis="both", labelsize=10)
                    if percentage:
                        ax2.set_xlabel("Mean(|SHAP Value|) (%)", fontsize=12)
                    else:
                        ax2.set_xlabel("Mean(|SHAP Value|)", fontsize=12)

        def conditional(data, gumshap, plot_file_path):
            """
            # NOTE: Overrides the default expl.ShapValues.conditional method to customize the visualization of SHAP values.
            Computes conditional SHAP values using the provided gumshap object, plots the results, and optionally saves the plot to a file.
            Args:
                data (pd.DataFrame): The input data for which to compute conditional SHAP values.
                gumshap: An object with a `_conditional` method for computing SHAP values and plotting capabilities.
                plot_file_path (str or None): The file path to save the plot. If None, the plot is displayed instead.
            Returns:
                dict: A dictionary mapping each column name to the mean absolute conditional SHAP value.
            Side Effects:
                - Generates and displays or saves a plot of the SHAP values.
                - Closes the current matplotlib figure after plotting.
            """

            results, v = gumshap._conditional(data)
            res = {}

            for col in results.columns:
                res[col] = abs(results[col].astype(float)).mean()

            _plotResults(gumshap, results, v)  # type: ignore library

            if plot_file_path is not None:
                plt.savefig(plot_file_path)
            else:
                plt.show()
            plt.close()
            return res

        # RFE: Calculate it only for the Markov Blanket of the target
        gumshap = expl.ShapValues(self.graphic, target)
        node_shapvalue_dict = conditional(data, gumshap, plot_file_path)

        graph = shapley.getShapValues(self.graphic, node_shapvalue_dict)
        graph.write_pdf(graph_file_path)

        return node_shapvalue_dict, graph

    def entropy_graph(self, graph_file_path: Path | None = None) -> pydot.Dot:
        """
        Generates and optionally saves an entropy information graph for the current Bayesian network.

        This method computes the information graph using the `expl.getInformationGraph` function,
        which provides information-theoretic values such as minimum and maximum information and mutual information.
        If a file path is provided, the resulting graph is saved as a PDF.

        Args:
            graph_file_path (Path | None, optional): The file path where the generated graph PDF should be saved.
                If None, the graph is not saved to disk.

        Returns:
            pydot.Dot: The generated information graph object.

        Notes
        -----
        - Blue node: Highest entropy (most uncertain).
        - Red node: Lowest entropy (least uncertain).
        - Arrow thickness: Represents the strength of mutual information between nodes.
        """

        result = expl.getInformationGraph(self.graphic, withMinMax=True)
        if isinstance(result, tuple):
            (
                graph,
                min_information_value,
                max_information_value,
                min_mutual_information_value,
                max_mutual_information_value,
            ) = result
            print(
                f"Min information value: {min_information_value}\n"
                f"Max information value: {max_information_value}\n"
                f"Min mutual information value: {min_mutual_information_value}\n"
                f"Max mutual information value: {max_mutual_information_value}"
            )
        else:
            graph = result
        if graph_file_path:
            graph.write_pdf(graph_file_path)
        return graph

    def conditional_independence_test(
        self,
        data: pd.DataFrame,
        target: str | None = None,
        plot_file_path: Path | None = None,
    ) -> dict[tuple[str, str, tuple[str]], float]:
        """
        Performs conditional independence tests for pairs of variables in the given DataFrame.
        This method computes p-values for conditional independence between pairs of variables,
        optionally with respect to a target variable.
        The results are returned as a dictionary mapping variable pairs (and conditioning sets)
        to their corresponding p-values. Optionally, the method can save a plot of the results to a specified file path.
        Args:
            data (pd.DataFrame): The input data containing variables to test for conditional independence.
            target (str | None, optional): An optional target variable to condition on. Defaults to None.
            plot_file_path (Path | None, optional): If provided, saves a plot of the results to this file path. Defaults to None.
        Returns:
            dict[tuple[str, str, tuple[str]], float]: A dictionary where keys are tuples representing
                (variable1, variable2, conditioning_set), and values are the corresponding p-values
                from the conditional independence tests.
        Notes
        -----
        - High p-values (typically > 0.05) might be conditionally independency. Suggest that the null hypothesis of independence cannot be rejected. There's insufficient evidence to conclude the variables are dependent
        - Low p-values (typically < 0.05) strong evidence of conditionally dependency. Suggest rejection of the independence hypothesis
        """
        # RFE: \perp warnings
        ci_pvalue_dict = expl.independenceListForPairs(
            self.graphic, data, target=target
        )
        if plot_file_path:
            plt.tight_layout()  # Adjust layout to prevent cutoff
            plt.savefig(plot_file_path, bbox_inches="tight", dpi=300)
        return ci_pvalue_dict

    def counterfactual(
        self,
        profile: dict[str, str | int | float],
        on: str | set[str],
        whatif: str | set[str],
        values: dict[str, str | int | float] | None = None,
    ) -> gum.Tensor:
        # TODO: For meaningful counterfactual inference about specific individuals, unobserved variables are essential to capture individual heterogeneity.
        def distribution_mean(pot: gum.Tensor) -> float | np.ndarray:
            """
            Calculates the mean (expected value) of a discrete probability distribution represented by a pyAgrum Tensor.
            Args:
                pot (gum.Tensor): A pyAgrum Tensor representing the probability distribution. The first variable of the tensor is assumed to be the random variable of interest.
            Returns:
                float | np.ndarray: The mean (expected value) of the distribution. Returns a float if the result is scalar, or a numpy ndarray if the result is multidimensional.
            Notes:
                - Assumes that `pot.variable(0).numerical(i)` returns the numerical value associated with the i-th state of the variable.
                - The mean is computed as the sum over all possible states: sum(value_i * probability_i).
            """

            return sum(
                [
                    pot.variable(0).numerical(i) * pot[i]
                    for i in range(pot.variable(0).domainSize())
                ]
            )

        """
        Computes the counterfactual outcome for a given profile under specified interventions.
        Args:
            profile (dict[str, str | int | float]): The observed values for the variables in the model.
            on (str | set[str]): The variable(s) to observe or condition on.
            whatif (str | set[str]): The variable(s) to intervene on (i.e., set to new values).
            values (dict[str, str | int | float]): The new values for the variables specified in `whatif`. If None, all possible values are considered.
        Returns:
            gum.Tensor: The resulting tensor representing the counterfactual distribution or outcome.
        Raises:
            Any exception raised by the underlying causal model or counterfactual computation.
        Example:
            >>> bn.counterfactual(
            ...    profile={"education": 0, "experience": 8, "salary": "86"},
            ...    on={"salary"},
            ...    whatif={"education"},
            ...    values={"education": 1},
            ... )
        """
        if values is not None:
            if isinstance(whatif, str):
                whatif = {whatif}
            missing_keys = set(values.keys()) - whatif
            if missing_keys:
                raise ValueError(
                    f"Keys {missing_keys} in 'values' are not present in 'whatif' variables."
                )
        cm = csl.CausalModel(self.graphic, latentVarsDescriptor=[], keepArcs=False)

        pot = csl.counterfactual(
            cm=cm,
            profile=profile,
            on=on,
            whatif=whatif,
            values=values,
        )
        # gnb.showProba(pot)
        # gnb.showInference(
        #     pot.observationalBN(), size="10", evs={"education": 0, "salary": "86"}
        # )
        # distribution_mean(pot)
        return pot

    def save(self, model_file: Path) -> None:
        """Saves the model to a file

        Args:
            model_file (Path): The model file path
        """
        super().save(model_file)
        gumimage.exportInference(self.graphic, str(model_file.with_suffix(".pdf")))

    def _fit_structure(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pbn.BayesianNetwork:
        if y is not None:
            # If y is provided, we add it to the data
            data = pd.concat([X, y], axis=1)
        else:
            data = X
        # Create a Bayesian network learner
        learner = gum.BNLearner(data)
        # learner.useSmoothingPrior(0.0)
        # learner.useGreedyHillClimbing()  # Use Greedy Hill Climbing algorithm

        # Add forbidden and mandatory arcs
        for source, target in self.arc_blacklist:
            learner.addForbiddenArc(source, target)
        for source, target in self.arc_whitelist:
            learner.addMandatoryArc(source, target)

        # Learn the structure and parameters
        self.graphic = learner.learnBN()
        bn = pbn.DiscreteBN(
            nodes=list(self.graphic.names()), arcs=convert_arcs_to_names(self.graphic)  # type: ignore library
        )

        return bn
