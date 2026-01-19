import numpy as np
import pandas as pd
import pybnesian as pbn
from src.spbnclassify.utils.constants import TRUE_ANOMALY_LABEL

from .base import BayesianNetwork


class GaussianBayesianNetwork(
    BayesianNetwork, pbn.CLGNetwork
):  # Method Resolution Order important (save/load)
    """Gaussian Bayesian Network class
    Gaussian log-likelihood formula:
    logl = -0.5 * ((x - mean) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))

    Standard Gaussian log-likelihood formula:
    logl = -np.log(2 * np.pi) / 2 - z**2 / 2
    """

    bn_type = pbn.CLGNetworkType()
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
    ) -> None:
        """Initializes the Gaussian Bayesian Network with the nodes, arcs, node_types and the structure learning parameters

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
        # if shortened_names:
        #     # Shortens the names of the nodes and arcs to the first letter of each node in uppercase
        #     nodes = [
        #         "".join(word[0] for word in node.split("_")).upper() for node in nodes
        #     ]
        #     arcs = [
        #         (
        #             "".join(word[0] for word in arc[0].split("_")).upper(),
        #             "".join(word[0] for word in arc[1].split("_")).upper(),
        #         )
        #         for arc in arcs
        #     ]
        # NOTE: This has to be called first to avoid errors, but __init__ doesn't rewrite nodes and arcs
        pbn.CLGNetwork.__init__(self, nodes=[])
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
        self.joint_gaussian_ = {"mean": None, "cov": None}

    def __str__(self) -> str:
        """Returns the string representation of the Gaussian Bayesian Network

        Returns:
            str: The string representation
        """
        return "Gaussian " + BayesianNetwork.__str__(self)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "BayesianNetwork":
        """Learns the structure and parameters of the Gaussian Bayesian Network from the data.

        Args:
            data (pd.DataFrame): The data to learn from.
        """
        data = pd.concat([X, y], axis=1)
        BayesianNetwork.fit(self, X, y)
        pbn.CLGNetwork.fit(self, data)
        self._calculate_max_logl(X)
        self.joint_gaussian_ = self._get_joint_gaussian()
        return self

    def _get_joint_gaussian(self) -> dict[str, pd.DataFrame]:
        """Calculates the equivalent joint Gaussian distribution of the Bayesian Network

        Returns:
            dict[str, pd.DataFrame]: The mean and covariance matrices of the joint Gaussian distribution
        """
        # The nodes have to calculate the values sequentially
        sorted_nodes = self.graph().topological_sort()

        # Initializes the mean and covariance matrices with ones for further multiplications. They are row vectors
        joint_mean = pd.DataFrame(
            1, index=np.arange(0, 1), columns=(["parent_0"] + self.nodes())
        )
        joint_cov = pd.DataFrame(
            1, index=self.nodes(), columns=self.nodes(), dtype=float
        )

        for i, node in enumerate(sorted_nodes):
            cpd = self.cpd(node)
            node_parents = ["parent_0"] + cpd.evidence()

            # Calculates the mean components
            joint_mean[node] = cpd.beta.T.dot(joint_mean[node_parents].T)

            # Calculates the variance components
            joint_cov.loc[node, node] = cpd.variance
            if len(node_parents) > 1:
                joint_cov.loc[node, node] += (
                    cpd.beta[1:]
                    .T.dot(joint_cov.loc[node_parents[1:], node_parents[1:]])
                    .dot(cpd.beta[1:])
                )

            # Calculates the covariance components
            for j in range(0, i):
                node_j = sorted_nodes[j]
                joint_cov.loc[node_j, node] = cpd.beta[1:].T.dot(
                    joint_cov.loc[node_j, node_parents[1:]]
                )
                joint_cov.loc[node, node_j] = joint_cov.loc[node_j, node]
        joint_mean.drop(columns=["parent_0"], inplace=True)
        return {"mean": joint_mean, "cov": joint_cov}
