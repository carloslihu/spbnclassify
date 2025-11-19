import pybnesian as pbn

from .banc import (
    BaseBayesianNetworkClassifier,
    GaussianBayesianNetworkAugmentedNaiveBayes,
    KDEBayesianNetworkAugmentedNaiveBayes,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
)


class MaxKAugmentedNaiveBayes(BaseBayesianNetworkClassifier):
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
            self.max_indegree = 1
        return super()._init_structure(nodes)

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network Classifier

        Returns:
            str: The string representation
        """
        return "Max-k " + super().__str__()


class GaussianMaxKAugmentedNaiveBayes(
    MaxKAugmentedNaiveBayes, GaussianBayesianNetworkAugmentedNaiveBayes
):
    pass


class SemiParametricMaxKAugmentedNaiveBayes(
    MaxKAugmentedNaiveBayes, SemiParametricBayesianNetworkAugmentedNaiveBayes
):
    pass


class KDEMaxKAugmentedNaiveBayes(
    MaxKAugmentedNaiveBayes, KDEBayesianNetworkAugmentedNaiveBayes
):
    pass
