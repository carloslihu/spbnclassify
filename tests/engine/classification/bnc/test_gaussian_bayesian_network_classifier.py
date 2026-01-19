import pandas as pd
import pybnesian as pbn
from helpers.data import SUPER_PARENT, TRUE_CLASS_LABEL
from src.spbnclassify.bnc import (
    GaussianAveragedOneDependenceEstimator,
    GaussianBayesianMultinet,
    GaussianBayesianNetworkAugmentedNaiveBayes,
    GaussianKDependenceBayesian,
    GaussianMaxKAugmentedNaiveBayes,
    GaussianNaiveBayes,
    GaussianSelectiveNaiveBayes,
    GaussianSuperParentOneDependenceEstimator,
    GaussianTreeAugmentedNaiveBayes,
)

# TODO: Remove rutile_ai dependency
from src.spbnclassify.utils.constants import TRUE_CLASS_LABEL

from .test_bayesian_network_classifier import BaseTestGaussianBayesianNetworkClassifier


class TestGaussianBayesianNetworkAugmentedNaiveBayes(
    BaseTestGaussianBayesianNetworkClassifier
):
    bn_class = GaussianBayesianNetworkAugmentedNaiveBayes
    model_filename = "gbnc.pkl"
    str_representation = "Gaussian Bayesian Augmented Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
            + [("c", "b")]
        )


class TestGaussianNaiveBayes(BaseTestGaussianBayesianNetworkClassifier):
    bn_class = GaussianNaiveBayes
    model_filename = "gnbnc.pkl"
    str_representation = "Naive Gaussian Bayesian Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
        )


class TestGaussianSelectiveNaiveBayes(BaseTestGaussianBayesianNetworkClassifier):
    bn_class = GaussianSelectiveNaiveBayes
    model_filename = "gsnbnc.pkl"
    str_representation = "Selective Naive Gaussian Bayesian Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        return {(TRUE_CLASS_LABEL, "b"), (TRUE_CLASS_LABEL, "c")}


class TestGaussianTreeAugmentedNaiveBayes(BaseTestGaussianBayesianNetworkClassifier):
    bn_class = GaussianTreeAugmentedNaiveBayes
    model_filename = "gtanbnc.pkl"
    str_representation = "Tree Augmented Naive Gaussian Bayesian Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
            + [("b", "c"), ("c", "a")]
        )


class TestGaussianSuperParentOneDependenceEstimator(
    BaseTestGaussianBayesianNetworkClassifier
):
    bn_class = GaussianSuperParentOneDependenceEstimator
    model_filename = "gspodbnc.pkl"
    str_representation = (
        "Superparent-one-dependence Gaussian Bayesian Network Classifier"
    )
    init_params = {"super_parent_": SUPER_PARENT}

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.drop(columns=[TRUE_CLASS_LABEL]).columns
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes]
            + [(SUPER_PARENT, node) for node in nodes if node != SUPER_PARENT]
        )


class TestGaussianAveragedOneDependenceEstimator(
    BaseTestGaussianBayesianNetworkClassifier
):
    bn_class = GaussianAveragedOneDependenceEstimator
    model_filename = "gaodebnc.pkl"
    str_representation = "Averaged one-dependence Gaussian Bayesian Network Classifier"
    init_params = {}

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        # Multinet doesn't have fixed arc structure
        return {
            "a": [
                (TRUE_CLASS_LABEL, "a"),
                (TRUE_CLASS_LABEL, "b"),
                (TRUE_CLASS_LABEL, "c"),
                ("a", "b"),
                ("a", "c"),
            ],
            "b": [
                (TRUE_CLASS_LABEL, "a"),
                (TRUE_CLASS_LABEL, "b"),
                (TRUE_CLASS_LABEL, "c"),
                ("b", "c"),
                ("b", "a"),
            ],
            "c": [
                (TRUE_CLASS_LABEL, "a"),
                (TRUE_CLASS_LABEL, "b"),
                (TRUE_CLASS_LABEL, "c"),
                ("c", "a"),
                ("c", "b"),
            ],
        }

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        """Override this method to define expected node types for the classifier."""
        return {
            "a": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.LinearGaussianCPDType(),
                "b": pbn.LinearGaussianCPDType(),
                "c": pbn.LinearGaussianCPDType(),
            },
            "b": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.LinearGaussianCPDType(),
                "b": pbn.LinearGaussianCPDType(),
                "c": pbn.LinearGaussianCPDType(),
            },
            "c": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.LinearGaussianCPDType(),
                "b": pbn.LinearGaussianCPDType(),
                "c": pbn.LinearGaussianCPDType(),
            },
        }


class TestGaussianKDependenceBayesian(BaseTestGaussianBayesianNetworkClassifier):
    bn_class = GaussianKDependenceBayesian
    model_filename = "gkdbnc.pkl"
    str_representation = "k-Dependence Gaussian Bayesian Network Classifier"
    init_params = {"max_indegree": 1}

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
            + [("c", "a"), ("c", "b"), ("b", "a")]
        )

    def has_max_indegree_constraint(self) -> bool:
        return True


class TestGaussianMaxKAugmentedNaiveBayes(BaseTestGaussianBayesianNetworkClassifier):
    bn_class = GaussianMaxKAugmentedNaiveBayes
    model_filename = "gmkbnc.pkl"
    str_representation = "Max-k Gaussian Bayesian Network Classifier"
    init_params = {"max_indegree": 1}

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
            + [("c", "b")]
        )

    def has_max_indegree_constraint(self) -> bool:
        return True


class TestGaussianBayesianMultinet(BaseTestGaussianBayesianNetworkClassifier):
    bn_class = GaussianBayesianMultinet
    model_filename = "gbmc.pkl"
    str_representation = "Gaussian Bayesian Multinet Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        # Multinet doesn't have fixed arc structure
        return {"class1": [("b", "c")], "class2": [("b", "c")], "class3": [("b", "c")]}

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        return {
            "class1": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.LinearGaussianCPDType(),
                "b": pbn.LinearGaussianCPDType(),
                "c": pbn.LinearGaussianCPDType(),
            },
            "class2": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.LinearGaussianCPDType(),
                "b": pbn.LinearGaussianCPDType(),
                "c": pbn.LinearGaussianCPDType(),
            },
            "class3": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.LinearGaussianCPDType(),
                "b": pbn.LinearGaussianCPDType(),
                "c": pbn.LinearGaussianCPDType(),
            },
        }
