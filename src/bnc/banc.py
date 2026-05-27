import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyagrum.clg as gclg
import pyagrum.clg.notebook as gclgnb
import pybnesian as pbn
from scipy.stats import norm

from ..bn import (
    GaussianBayesianNetwork,
    KDEBayesianNetwork,
    SemiParametricBayesianNetwork,
)
from ..utils.constants import (
    CONTINUOUS_NODES,
    PROB_CONTINUOUS_CONTINUOUS,
    PROB_DISCRETE_CONTINUOUS,
    PROB_DISCRETE_DISCRETE,
    PROB_GAUSSIAN,
    TRUE_CLASS_LABEL,
)
from ..utils.generic import safe_exp
from .base import BaseBayesianNetworkClassifier
from .probabilistic_model import FixedCLG, FixedDiscreteFactor, NormalMixtureCPD


class BayesianNetworkAugmentedNaiveBayes(BaseBayesianNetworkClassifier):
    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network Classifier

        Returns:
            str: The string representation
        """
        return "Augmented " + super().__str__()

    # def fit(
    #     self, X: pd.DataFrame, y: pd.Series | None = None
    # ) -> "BayesianNetworkAugmentedNaiveBayes":
    #     super().fit(X, y)
    #     # RFE: Calculate with hybrid factors
    #     # self._calculate_max_logl(data)
    #     # RFE: Calculate joint_gaussian?
    #     # self.joint_gaussian_ = self._get_joint_gaussian()
    #     return self


class GaussianBayesianNetworkAugmentedNaiveBayes(
    BayesianNetworkAugmentedNaiveBayes, GaussianBayesianNetwork
):

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
        true_label: str = TRUE_CLASS_LABEL,
        prediction_label: str = "predicted_label",
        classes_: list[str] = [],
        weights_: dict[str, float] = {},
    ) -> None:
        # Redundant but more sustainable
        GaussianBayesianNetwork.__init__(
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
        BayesianNetworkAugmentedNaiveBayes.__init__(
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
            classes_=classes_,
            weights_=weights_,
        )
        self.graphic_dict = {}

    def __str__(self) -> str:
        """Returns the string representation of the Gaussian Bayesian Network

        Returns:
            str: The string representation
        """
        return "Gaussian " + BayesianNetworkAugmentedNaiveBayes.__str__(self)

    def _class_posterior_from_evidence(
        self, evidence: dict[str, float]
    ) -> dict[str, float]:
        """Compute P(C | E) for every class using Bayes theorem.

        The implementation builds a log-likelihood matrix where each row is a class and
        each column is an evidence variable. The posterior is then computed as:

            P(C = k | E) = P(C = k) * ∏_m P(E_m | C = k) / P(E)
        """
        # log P(C = k) is given by the class weights
        log_scores = np.log(self.weights_)

        for class_index, class_value in enumerate(self.classes_):
            bn = self.graphic_dict[class_value]
            for variable_name, observed_value in evidence.items():
                variable = bn.variable(variable_name)
                # log P(E_m | C = k)
                log_prob_e_given_c = norm.logpdf(
                    observed_value, loc=variable.mu(), scale=variable.sigma()
                )
                log_scores[class_index] += log_prob_e_given_c

        posterior = safe_exp(log_scores)
        posterior_sum = posterior.sum()
        posterior /= posterior_sum

        return dict(zip(self.classes_, posterior))

    def infer(
        self,
        evidence: dict[str, float] = {},
        json_file_path: Path | None = None,
        pdf_file_path: Path | None = None,
    ) -> dict[str, dict]:
        """
        Performs inference on the Bayesian network using the provided evidence and target nodes.
        Args:
            evidence (dict[str, float], optional): A dictionary mapping node names to their observed values. Defaults to an empty dictionary. We can have hard evidence (e.g., {"Execution": True}) or soft evidence (e.g., {"Execution": [0.3, 0.9]}).
            json_file_path (Path | None, optional): If provided, exports the inference results to this file in JSON format.
            pdf_file_path (Path | None, optional): If provided, exports the graphical representation of the inference to this file in PDF format.
        Returns:
            dict[str, dict]: A dictionary where keys are node names and values are dictionaries containing the posterior probabilities for each state of the node.
        """
        infer_dict = {}
        infer_dict["structure"] = list(self.graphic.arcs())
        infer_dict["parameters"] = {"evidence": evidence}
        # For each class-specific CLG, we perform inference and extract the posterior distribution for each variable.
        for class_value in self.classes_:
            infer_dict["parameters"][class_value] = {}
            bn = self.graphic_dict[class_value]
            ie = gclg.CLGVariableElimination(bn)
            ie.updateEvidence(evidence)

            for variable_name in bn.names():
                # If the variable is in the evidence, we directly use the evidence value as the posterior.
                if variable_name in evidence:
                    post = gclg.GaussianVariable(
                        variable_name, evidence[variable_name], 0
                    )
                # If the variable is not in the evidence, we perform inference to get the posterior distribution.
                else:
                    posterior_cf = ie.canonicalPosterior([variable_name])
                    # If the posterior is a Gaussian, we extract the mean and variance to create a GaussianVariable. If the posterior is a scalar (which can happen in disconnected graphs), we directly get the variable from the graphic.
                    if hasattr(posterior_cf, "toGaussian"):
                        _, mu, var = posterior_cf.toGaussian()
                        post = gclg.GaussianVariable(variable_name, mu, np.sqrt(var))
                    else:
                        post = self.graphic_dict[class_value].variable(variable_name)
                mu = post.mu()
                std = post.sigma()
                infer_dict["parameters"][class_value][variable_name] = {
                    "variable_name": variable_name,
                    "probabilities": {"mean": mu, "std": std},
                }
        prob_c_given_e = self._class_posterior_from_evidence(evidence)
        for class_value in self.classes_:
            infer_dict["parameters"][class_value]["prob_c_given_e"] = prob_c_given_e[
                class_value
            ]

        # export results
        if json_file_path:
            with open(json_file_path, "w") as f:
                json.dump(infer_dict, f, indent=4)
        # TODO: Fix bug with export inference because of sparse graph?
        # if pdf_file_path:
        #     for class_value in self.classes_:
        #         output_file = pdf_file_path.with_name(
        #             f"{pdf_file_path.stem}_{class_value}.pdf"
        #         )
        #         gclgnb.exportInference(
        #             clg=self.graphic_dict[class_value],
        #             filename=str(output_file),
        #             evs=evidence,
        #         )

        return infer_dict

    def posterior(
        self,
        query_var: str,
        evidence: dict[str, float],
        point: pd.Series,
    ) -> float:
        """
        Compute the posterior probability of a query variable given evidence.

        This method calculates P(X | E) by marginalizing over the classes,
        using the posterior distributions obtained from inference.

        Args:
            query_var: The variable to query (must be a subset of graph nodes).
            evidence: Dictionary of evidence variables and their values (must be subset of graph nodes).
            point: A pandas Series containing the point value for the query variable to evaluate.

        Returns:
            float: The posterior probability P(X | E) evaluated at the given point.

        Raises:
            ValueError: If query_var or evidence variables are not in the graph,
                or if query_var and evidence share common variables.
        """
        if not set(query_var).issubset(set(self.nodes())):
            raise ValueError(
                "Query variables must be a subset of the nodes in the graph."
            )
        if not set(evidence.keys()).issubset(set(self.nodes())):
            raise ValueError(
                "Evidence variables must be a subset of the nodes in the graph."
            )
        if set(evidence.keys()).intersection(set(query_var)):
            raise ValueError("Query variables and evidence variables must be disjoint.")
        if self.true_label in evidence:
            classes = [evidence[self.true_label]]
            # Remove self.true_label from evidence
            evidence = {k: v for k, v in evidence.items() if k != self.true_label}
        else:  # If the class variable is not in the evidence, we need to marginalize over the classes to compute P(X | E).
            classes = self.classes_

        infer_dict = self.infer(evidence=evidence)
        prob_x_given_e = 0
        for class_value in classes:
            prob_c_given_e = infer_dict["parameters"][class_value]["prob_c_given_e"]
            # for query_var in query_vars:
            variable_posterior = infer_dict["parameters"][class_value][query_var][
                "probabilities"
            ]
            mu = variable_posterior["mean"]
            std = variable_posterior["std"]
            # Calculate P(X | C, E) using the posterior distribution of the variable given the evidence. This is done by evaluating the Gaussian PDF at the point value for the variable.
            prob_x_given_c_e = norm.pdf(point[query_var], loc=mu, scale=std)
            # We can then calculate P(X | E) by marginalizing over the classes:
            # P(X | E) = ∑_k P(X | C = k, E) * P(C = k | E)
            prob_x_given_e += prob_x_given_c_e * prob_c_given_e
        return prob_x_given_e

    # TODO: Calculate inference value or the most probable explanation probability?

    # TODO: Implement GBNC specific method
    def _get_joint_gaussian(self) -> dict[str, pd.DataFrame]:
        return {}

    def _set_gum_params(self) -> None:
        # Copies the nodes to each class-specific pyagrum graphic
        for class_value in self.classes_:
            self.graphic_dict[class_value] = gclg.CLG()
            for node in self.nodes():
                cpd = self.conditional_factor(node, class_value)
                if cpd.type() == pbn.LinearGaussianCPDType():
                    mu = cpd.beta[0]
                    std = np.sqrt(cpd.variance)
                    var = gclg.GaussianVariable(node, mu, std)
                    self.graphic_dict[class_value].add(var)

            # Copies the arcs to the pyagrum graphic
            for source, target in self.arcs():
                if source != TRUE_CLASS_LABEL:
                    cpd = self.conditional_factor(target, class_value)
                    parents = cpd.evidence()
                    parent_index = parents.index(source)
                    coef = cpd.beta[parent_index + 1]
                    self.graphic_dict[class_value].addArc(source, target, coef)


class SemiParametricBayesianNetworkAugmentedNaiveBayes(
    BayesianNetworkAugmentedNaiveBayes, SemiParametricBayesianNetwork
):  # Method Resolution Order important (save/load)

    def __init__(
        self,
        search_score: str = "validated-lik",
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
        true_label: str = TRUE_CLASS_LABEL,
        prediction_label: str = "predicted_label",
        classes_: list[str] = [],
        weights_: dict[str, float] = {},
    ) -> None:
        """Initializes the SemiParametric Bayesian Network with the nodes, arcs, node_types and the structure learning parameters

        Args:
            nodes (list[str], optional): list of nodes. Defaults to [].
            arcs (list[tuple[str, str]], optional): list of arcs. Defaults to [].
            node_types (list[tuple[str, pbn.FactorType]], optional): list of node types. Defaults to [].
            search_score (str): Search score to be used for the structure learning. The possible scores ((validate_options.cpp)) are:
                - "cv-lik" (Cross-Validated likelihood)
                - "holdout-lik" (Hold-out likelihood)
                - "validated-lik" (Validated likelihood with cross-validation). Defaults to "validated-lik".
            arc_blacklist (list[tuple[str, str]], optional): Arc blacklist (forbidden arcs). Defaults to [].
            arc_whitelist (list[tuple[str, str]], optional): Arc whitelist (forced arcs). Defaults to [].
            type_blacklist (list[tuple[str, pbn.FactorType]], optional): Node type blacklist (forbidden node types). Defaults to [].
            type_whitelist (list[tuple[str, pbn.FactorType]], optional): Node type whitelist (forced node types). Defaults to [].
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
        SemiParametricBayesianNetwork.__init__(
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
        BayesianNetworkAugmentedNaiveBayes.__init__(
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
            classes_=classes_,
            weights_=weights_,
        )

    def __str__(self) -> str:
        """Returns the string representation of the SemiParametric Bayesian Network

        Returns:
            str: The string representation
        """
        return "SemiParametric " + BayesianNetworkAugmentedNaiveBayes.__str__(self)

    # TODO: Add method to bn.base and other parametric and nonparametric families
    # TODO: Add tests
    @classmethod
    def generate(cls, seed) -> BayesianNetworkAugmentedNaiveBayes:
        bn = cls(
            feature_names_in_=CONTINUOUS_NODES,
            true_label=TRUE_CLASS_LABEL,
        )
        bn._generate_structure(seed=seed)
        bn._generate_parameters(seed=seed)
        return bn

    def _generate_structure(
        self,
        seed: int | None = None,
    ) -> None:
        np.random.seed(seed)
        # We define a discrete node and 5 continuous nodes

        # Generate random node types
        discrete_nodes = [self.true_label]
        discrete_node_types = [(n, pbn.DiscreteFactorType()) for n in discrete_nodes]

        continuous_node_types = list(
            zip(
                self.feature_names_in_,
                np.random.choice(
                    [pbn.LinearGaussianCPDType(), pbn.CKDEType()],
                    size=len(self.feature_names_in_),
                    p=[
                        PROB_GAUSSIAN,
                        1 - PROB_GAUSSIAN,
                    ],
                ),
            )
        )
        node_types = discrete_node_types + continuous_node_types

        # Generate arcs between discrete nodes
        arcs = np.asarray(
            [
                (s, d)
                for i, s in enumerate(discrete_nodes[:-1])
                for d in discrete_nodes[i + 1 :]
            ]
        )
        num_arcs = arcs.shape[0]
        active_arcs_index = np.random.choice(
            2, size=num_arcs, p=[1 - PROB_DISCRETE_DISCRETE, PROB_DISCRETE_DISCRETE]
        )

        discrete_arcs = list(arcs[active_arcs_index == 1])

        # Generate arcs between discrete nodes and continuous nodes
        arcs = np.asarray(
            [(s, d) for s in discrete_nodes for d in self.feature_names_in_]
        )
        num_arcs = arcs.shape[0]
        active_arcs_index = np.random.choice(
            2, size=num_arcs, p=[1 - PROB_DISCRETE_CONTINUOUS, PROB_DISCRETE_CONTINUOUS]
        )

        discrete_continuous_arcs = list(arcs[active_arcs_index == 1])

        # Generate arcs between continuous nodes
        arcs = np.asarray(
            [
                (s, d)
                for i, s in enumerate(self.feature_names_in_[:-1])
                for d in self.feature_names_in_[i + 1 :]
            ]
        )
        num_arcs = arcs.shape[0]
        active_arcs_index = np.random.choice(
            2,
            size=num_arcs,
            p=[1 - PROB_CONTINUOUS_CONTINUOUS, PROB_CONTINUOUS_CONTINUOUS],
        )

        continuous_arcs = list(arcs[active_arcs_index == 1])

        arcs = discrete_arcs + discrete_continuous_arcs + continuous_arcs

        self._copy_bn_structure(arcs=arcs, node_types=node_types)

    def _generate_parameters(
        self,
        seed: int | None = None,
    ) -> None:
        np.random.seed(seed)
        discrete_nodes = [self.true_label]
        cpds = []

        for node in self.nodes():
            node_type = self.node_type(node)
            parents = self.parents(node)
            if node_type == pbn.DiscreteFactorType():
                cpd = FixedDiscreteFactor.new_random_cpd(node, parents)
            else:  # Continuous nodes
                discrete_evidence = list(set(parents).intersection(set(discrete_nodes)))
                continuous_evidence = list(
                    set(parents).intersection(set(self.feature_names_in_))
                )
                if node_type == pbn.LinearGaussianCPDType():
                    cpd = FixedCLG.new_random_cpd(
                        node, discrete_evidence, continuous_evidence
                    )
                elif node_type == pbn.CKDEType():
                    cpd = NormalMixtureCPD.new_random_cpd(
                        node, discrete_evidence, continuous_evidence
                    )
                else:
                    raise ValueError(f"Unknown node type: {node_type}.")
            cpds.append(cpd)
        self.add_cpds(cpds)


class KDEBayesianNetworkAugmentedNaiveBayes(
    SemiParametricBayesianNetworkAugmentedNaiveBayes, KDEBayesianNetwork
):  # Method Resolution Order important (save/load)

    def __init__(
        self,
        search_score: str = "validated-lik",
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
        true_label: str = TRUE_CLASS_LABEL,
        prediction_label: str = "predicted_label",
        classes_: list[str] = [],
        weights_: dict[str, float] = {},
    ) -> None:
        """Initializes the KDE Bayesian Network with the nodes, arcs, node_types and the structure learning parameters

        Args:
            nodes (list[str], optional): list of nodes. Defaults to [].
            arcs (list[tuple[str, str]], optional): list of arcs. Defaults to [].
            node_types (list[tuple[str, pbn.FactorType]], optional): list of node types. Defaults to [].
            search_score (str): Search score to be used for the structure learning. The possible scores ((validate_options.cpp)) are:
                - "cv-lik" (Cross-Validated likelihood)
                - "holdout-lik" (Hold-out likelihood)
                - "validated-lik" (Validated likelihood with cross-validation). Defaults to "validated-lik".
            arc_blacklist (list[tuple[str, str]], optional): Arc blacklist (forbidden arcs). Defaults to [].
            arc_whitelist (list[tuple[str, str]], optional): Arc whitelist (forced arcs). Defaults to [].
            type_blacklist (list[tuple[str, pbn.FactorType]], optional): Node type blacklist (forbidden node types). Defaults to [].
            type_whitelist (list[tuple[str, pbn.FactorType]], optional): Node type whitelist (forced node types). Defaults to [].
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
        KDEBayesianNetwork.__init__(
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
        SemiParametricBayesianNetworkAugmentedNaiveBayes.__init__(
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
            classes_=classes_,
            weights_=weights_,
        )

    def _init_structure(
        self,
        nodes: list[str] = [],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, pbn.FactorType]]]:
        """
        Initializes the structure of the Bayesian network classifier.

        This method generates the arcs (directed edges) and node types for the
        Bayesian network based on the provided list of nodes. The arcs connect
        the `true_label` node to all other nodes, and the node types specify
        the factor type for each node.

        Args:
            nodes (list[str]): A list of node names to include in the network.
                The `true_label` node is treated as a discrete factor, while
                all other nodes are treated as CKDE factors.

        Returns:
            tuple[list[tuple[str, str]], list[tuple[str, pbn.FactorType]]]:
                A tuple containing:
                - A list of arcs, where each arc is represented as a tuple
                (parent, child).
                - A list of node types, where each node type is represented
                as a tuple (node_name, factor_type).
        """
        arcs = [(self.true_label, node) for node in nodes if node != self.true_label]

        node_types = [(self.true_label, pbn.DiscreteFactorType())] + [
            (node, pbn.CKDEType()) for node in nodes if node != self.true_label
        ]

        return arcs, node_types

    def __str__(self) -> str:
        """Returns the string representation of the KDE Bayesian Network

        Returns:
            str: The string representation
        """
        return "KDE " + BayesianNetworkAugmentedNaiveBayes.__str__(self)
