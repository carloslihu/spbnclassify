import pandas as pd
import pybnesian as pbn
from helpers.data import SEED, SUPER_PARENT, TRUE_CLASS_LABEL
from src.spbnclassify.bnc import (
    BaseBayesianNetworkClassifier,
    KDEAveragedOneDependenceEstimator,
    KDEBayesianMultinet,
    KDEBayesianNetworkAugmentedNaiveBayes,
    KDEKDependenceBayesian,
    KDEMaxKAugmentedNaiveBayes,
    KDENaiveBayes,
    KDESelectiveNaiveBayes,
    KDESuperParentOneDependenceEstimator,
    KDETreeAugmentedNaiveBayes,
)

# TODO: Remove rutile_ai dependency
from src.spbnclassify.utils.constants import TRUE_CLASS_LABEL

from .test_bayesian_network_classifier import BaseTestKDEBayesianNetworkClassifier


class TestKDEBayesianNetworkAugmentedNaiveBayes(BaseTestKDEBayesianNetworkClassifier):
    bn_class = KDEBayesianNetworkAugmentedNaiveBayes
    model_filename = "kdebnc.pkl"
    str_representation = "KDE Bayesian Augmented Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
            + [("b", "c"), ("a", "c")]
        )

    def test_generate(self) -> None:
        """Test the generate method of the KDE Bayesian Network Classifier."""
        generated_arc_set = {
            ("a", "e"),
            (TRUE_CLASS_LABEL, "a"),
            ("c", "d"),
            ("a", "c"),
            (TRUE_CLASS_LABEL, "d"),
            (TRUE_CLASS_LABEL, "b"),
            ("a", "d"),
            ("d", "e"),
            ("a", "b"),
            (TRUE_CLASS_LABEL, "e"),
            ("c", "e"),
            (TRUE_CLASS_LABEL, "c"),
        }
        generated_bn = KDEBayesianNetworkAugmentedNaiveBayes.generate(seed=SEED)
        assert isinstance(generated_bn, BaseBayesianNetworkClassifier)
        assert generated_bn.num_nodes() == len(generated_bn.feature_names_in_) + 1
        assert set(generated_bn.arcs()) == generated_arc_set


class TestKDENaiveBayes(BaseTestKDEBayesianNetworkClassifier):
    bn_class = KDENaiveBayes
    model_filename = "kdenbnc.pkl"
    str_representation = "Naive KDE Bayesian Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
        )


class TestKDESelectiveNaiveBayes(BaseTestKDEBayesianNetworkClassifier):
    bn_class = KDESelectiveNaiveBayes
    model_filename = "kdesnbnc.pkl"
    str_representation = "Selective Naive KDE Bayesian Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        return {(TRUE_CLASS_LABEL, "b"), (TRUE_CLASS_LABEL, "c")}


class TestKDETreeAugmentedNaiveBayes(BaseTestKDEBayesianNetworkClassifier):
    bn_class = KDETreeAugmentedNaiveBayes
    model_filename = "kdetanbnc.pkl"
    str_representation = "Tree Augmented Naive KDE Bayesian Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
            + [("c", "b"), ("a", "c")]
        )


class TestKDESuperParentOneDependenceEstimator(BaseTestKDEBayesianNetworkClassifier):
    bn_class = KDESuperParentOneDependenceEstimator
    model_filename = "kdespodbnc.pkl"
    str_representation = "Superparent-one-dependence KDE Bayesian Network Classifier"
    init_params = {"super_parent": SUPER_PARENT}

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.drop(columns=[TRUE_CLASS_LABEL]).columns
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes]
            + [(SUPER_PARENT, node) for node in nodes if node != SUPER_PARENT]
        )


class TestKDEAveragedOneDependenceEstimator(BaseTestKDEBayesianNetworkClassifier):
    bn_class = KDEAveragedOneDependenceEstimator
    model_filename = "kdeaodebnc.pkl"
    str_representation = "Averaged one-dependence KDE Bayesian Network Classifier"
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
        return {
            "a": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.CKDEType(),
                "b": pbn.CKDEType(),
                "c": pbn.CKDEType(),
            },
            "b": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.CKDEType(),
                "b": pbn.CKDEType(),
                "c": pbn.CKDEType(),
            },
            "c": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.CKDEType(),
                "b": pbn.CKDEType(),
                "c": pbn.CKDEType(),
            },
        }


class TestKDEKDependenceBayesian(BaseTestKDEBayesianNetworkClassifier):
    bn_class = KDEKDependenceBayesian
    model_filename = "kdekdbnc.pkl"
    str_representation = "k-Dependence KDE Bayesian Network Classifier"
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


class TestKDEMaxKAugmentedNaiveBayes(BaseTestKDEBayesianNetworkClassifier):
    bn_class = KDEMaxKAugmentedNaiveBayes
    model_filename = "kdemaxkbnc.pkl"
    str_representation = "Max-k KDE Bayesian Network Classifier"
    init_params = {"max_indegree": 1}

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
            + [("c", "a")]
        )

    def has_max_indegree_constraint(self) -> bool:
        return True


class TestKDEBayesianMultinet(BaseTestKDEBayesianNetworkClassifier):
    bn_class = KDEBayesianMultinet
    model_filename = "kdebmc.pkl"
    str_representation = "KDE Bayesian Multinet Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        # Multinet doesn't have fixed arc structure
        return {
            "class1": [("b", "c"), ("a", "c")],
            "class2": [("b", "c"), ("a", "c")],
            "class3": [("b", "c"), ("a", "c")],
        }

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        return {
            "class1": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.CKDEType(),
                "b": pbn.CKDEType(),
                "c": pbn.CKDEType(),
            },
            "class2": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.CKDEType(),
                "b": pbn.CKDEType(),
                "c": pbn.CKDEType(),
            },
            "class3": {
                TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
                "a": pbn.CKDEType(),
                "b": pbn.CKDEType(),
                "c": pbn.CKDEType(),
            },
        }
