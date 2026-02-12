import numpy as np
import pandas as pd
import pybnesian as pbn

from ..bn import (
    GaussianBayesianNetwork,
    KDEBayesianNetwork,
    SemiParametricBayesianNetwork,
)
from ..utils import (
    CONTINUOUS_NODES,
    PROB_CONTINUOUS_CONTINUOUS,
    PROB_DISCRETE_CONTINUOUS,
    PROB_DISCRETE_DISCRETE,
    PROB_GAUSSIAN,
)
from ..utils.constants import TRUE_CLASS_LABEL
from .base import BaseBayesianNetworkClassifier
from .probabilistic_model import FixedCLG, FixedDiscreteFactor, NormalMixtureCPD


class BayesianNetworkAugmentedNaiveBayes(BaseBayesianNetworkClassifier):
    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network Classifier

        Returns:
            str: The string representation
        """
        return "Augmented " + super().__str__()

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "BayesianNetworkAugmentedNaiveBayes":
        if y is None:
            raise ValueError("y must be set")
        data = pd.concat([X, y], axis=1)
        nodes = data.columns.tolist()

        arcs, node_types = self._init_structure(nodes)
        # NOTE: reinit doesn't work, best to copy the structure
        # RFE: I should reset the structure each time before fit is called
        self._copy_bn_structure(arcs, node_types)
        super().fit(X, y)
        # RFE: Calculate with hybrid factors
        # self._calculate_max_logl(data)
        # RFE: Calculate joint_gaussian?
        # self.joint_gaussian_ = self._get_joint_gaussian()
        return self


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

    def __str__(self) -> str:
        """Returns the string representation of the Gaussian Bayesian Network

        Returns:
            str: The string representation
        """
        return "Gaussian " + BayesianNetworkAugmentedNaiveBayes.__str__(self)

    # TODO: Implement GBNC specific method
    def _get_joint_gaussian(self) -> dict[str, pd.DataFrame]:
        return {}


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

    # RFE: Refactor Inheritance
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
