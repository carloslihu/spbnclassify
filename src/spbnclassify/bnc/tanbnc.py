import pandas as pd
import pybnesian as pbn

from ..utils import ConditionalMutualInformationGraph, DirectedTree
from .banc import (
    BaseBayesianNetworkClassifier,
    GaussianBayesianNetworkAugmentedNaiveBayes,
    KDEBayesianNetworkAugmentedNaiveBayes,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
)


class TreeAugmentedNaiveBayes(BaseBayesianNetworkClassifier):
    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network Classifier

        Returns:
            str: The string representation
        """
        return "Tree Augmented Naive " + super().__str__()

    def _fit_structure(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pbn.BayesianNetwork:
        if y is None:
            raise ValueError("y must be set")
        mutual_info_graph = ConditionalMutualInformationGraph()
        mutual_info_graph.calculate_maximum_weighted_spanning_tree(X, y)
        # We obtain the directed tree structure
        edges = mutual_info_graph.result
        root = edges[0][0]  # Arbitrary root
        tree = DirectedTree(root=root, edges=edges)
        edges = tree.get_edges()
        for node, node2, _ in edges:
            self.add_arc(node, node2)

        return self


class GaussianTreeAugmentedNaiveBayes(
    TreeAugmentedNaiveBayes,
    GaussianBayesianNetworkAugmentedNaiveBayes,
):
    pass


class SemiParametricTreeAugmentedNaiveBayes(
    TreeAugmentedNaiveBayes,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
):
    pass


class KDETreeAugmentedNaiveBayes(
    TreeAugmentedNaiveBayes, KDEBayesianNetworkAugmentedNaiveBayes
):
    pass
