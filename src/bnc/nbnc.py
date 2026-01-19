import pandas as pd
import pybnesian as pbn

from .banc import (
    BaseBayesianNetworkClassifier,
    GaussianBayesianNetworkAugmentedNaiveBayes,
    KDEBayesianNetworkAugmentedNaiveBayes,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
)


class NaiveBayes(BaseBayesianNetworkClassifier):
    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network Classifier

        Returns:
            str: The string representation
        """
        return "Naive " + super().__str__()

    def _fit_structure(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pbn.BayesianNetwork:
        return self


class GaussianNaiveBayes(NaiveBayes, GaussianBayesianNetworkAugmentedNaiveBayes):
    pass


class SemiParametricNaiveBayes(
    NaiveBayes, SemiParametricBayesianNetworkAugmentedNaiveBayes
):
    pass


class KDENaiveBayes(NaiveBayes, KDEBayesianNetworkAugmentedNaiveBayes):
    pass
