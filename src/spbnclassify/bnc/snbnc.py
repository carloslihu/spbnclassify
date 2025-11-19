import pandas as pd
import pybnesian as pbn

from ..utils import ConditionalMutualInformationGraph
from .banc import (
    BaseBayesianNetworkClassifier,
    GaussianBayesianNetworkAugmentedNaiveBayes,
    KDEBayesianNetworkAugmentedNaiveBayes,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
)


class SelectiveNaiveBayes(BaseBayesianNetworkClassifier):
    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network Classifier

        Returns:
            str: The string representation
        """
        return "Selective Naive " + super().__str__()

    def _fit_structure(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pbn.BayesianNetwork:
        if y is None:
            raise ValueError("y must be set")
        # Calculate mutual information for each feature with the target variable
        mutual_info_graph = ConditionalMutualInformationGraph()

        mi_values = mutual_info_graph.calculate_class_mutual_info(X, y)
        for node, mi_value in mi_values.items():
            if mi_value == 0:  # RFE: customizable threshold (based on p-value?)
                self.remove_arc(self.true_label, node)
        return self


class GaussianSelectiveNaiveBayes(
    SelectiveNaiveBayes, GaussianBayesianNetworkAugmentedNaiveBayes
):
    pass


class SemiParametricSelectiveNaiveBayes(
    SelectiveNaiveBayes,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
):
    pass


class KDESelectiveNaiveBayes(
    SelectiveNaiveBayes, KDEBayesianNetworkAugmentedNaiveBayes
):
    pass
