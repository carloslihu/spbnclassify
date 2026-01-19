import pandas as pd
import pybnesian as pbn

from ..utils import ConditionalMutualInformationGraph
from .banc import (
    GaussianBayesianNetworkAugmentedNaiveBayes,
    KDEBayesianNetworkAugmentedNaiveBayes,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
)
from .base import BaseBayesianNetworkClassifier


class KDependenceBayesian(BaseBayesianNetworkClassifier):
    def _init_structure(
        self,
        nodes: list[str] = [],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, pbn.FactorType]]]:
        """
        Initializes the structure of the Bayesian network classifier.
        Args:
            nodes (list[str]): A list of node names to include in the network. Defaults to an empty list.
            max_indegree (int): The maximum number of incoming edges (parents) allowed for any node.
                Must be greater than 0.
        Returns:
            tuple: A tuple containing:
                - list[tuple[str, str]]: A list of edges represented as tuples of node names (parent, child).
                - list[tuple[str, pbn.FactorType]]: A list of factors associated with the nodes.
        Raises:
            ValueError: If `max_indegree` is not set (i.e., equals 0).
        """
        # Error control for when max_indegree is not set
        if self.max_indegree == 0:
            self.max_indegree = 2  # This is the default k (k=1 is TAN)
        return super()._init_structure(nodes)

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network Classifier

        Returns:
            str: The string representation
        """
        return "k-Dependence " + super().__str__()

    def _fit_structure(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pbn.BayesianNetwork:

        if y is None:
            raise ValueError("y must be set")
        mutual_info_graph = ConditionalMutualInformationGraph()
        # k-DB algorithm” ([Sahami, 1996, p. 337])
        # 1. Calculate I(X_i, C) for all nodes and order them by highest mutual information with the class
        class_mi_values = mutual_info_graph.calculate_class_mutual_info(X, y)
        sorted_nodes = sorted(
            class_mi_values, key=lambda node: class_mi_values[node], reverse=True
        )
        # 2. Calculate I(X_i; X_j | C) for all pairs of nodes and order them by highest mutual information
        mutual_info_edges = mutual_info_graph.calculate_conditional_mutual_info(X, y)
        sorted_edges = sorted(mutual_info_edges, key=lambda x: x[2], reverse=True)
        # 3. Let S be the set of used nodes
        S = {}
        for X_max in sorted_nodes:
            # 5.1-5.3 Select X_max where I(X_max, C) is maximized, then add node C -> X_max
            # if X_max not in S:
            # NOTE: Already added in _init_structure
            # self.add_arc(self.true_label, X_max)
            # 5.4 Add M = min(|S|, k) parents X_j to X_max where I(X_max, X_j | C) is maximized
            M = min(len(S), self.max_indegree)
            m = 0
            for u, v, _ in sorted_edges:
                # If we have added enough parents, break
                if m >= M:
                    break
                elif u == X_max and v in S:
                    self.add_arc(v, X_max)
                    m += 1
                elif v == X_max and u in S:
                    self.add_arc(u, X_max)
                    m += 1

            # 5.5. Add X_max to the set of used nodes
            S[X_max] = True

        return self


class GaussianKDependenceBayesian(
    KDependenceBayesian,
    GaussianBayesianNetworkAugmentedNaiveBayes,
):
    pass


class SemiParametricKDependenceBayesian(
    KDependenceBayesian,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
):
    pass


class KDEKDependenceBayesian(
    KDependenceBayesian, KDEBayesianNetworkAugmentedNaiveBayes
):
    pass
