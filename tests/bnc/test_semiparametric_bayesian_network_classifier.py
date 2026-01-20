import pandas as pd
import pybnesian as pbn
from helpers.data import SEED, SUPER_PARENT, TRUE_CLASS_LABEL
from src.bnc import (
    BaseBayesianNetworkClassifier,
    SemiParametricAveragedOneDependenceEstimator,
    SemiParametricBayesianMultinet,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
    SemiParametricKDependenceBayesian,
    SemiParametricMaxKAugmentedNaiveBayes,
    SemiParametricNaiveBayes,
    SemiParametricSelectiveNaiveBayes,
    SemiParametricSuperParentOneDependenceEstimator,
    SemiParametricTreeAugmentedNaiveBayes,
)

# TODO: Remove rutile_ai dependency
from src.utils.constants import TRUE_CLASS_LABEL

from .test_bayesian_network_classifier import (
    BaseTestSemiParametricBayesianNetworkClassifier,
)


class TestSemiParametricBayesianNetworkAugmentedNaiveBayes(
    BaseTestSemiParametricBayesianNetworkClassifier
):
    bn_class = SemiParametricBayesianNetworkAugmentedNaiveBayes
    model_filename = "spbnc.pkl"
    str_representation = "SemiParametric Bayesian Augmented Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
            + [("b", "c"), ("a", "c")]
        )

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        return {
            TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
            "a": pbn.CKDEType(),
            "b": pbn.CKDEType(),
            "c": pbn.LinearGaussianCPDType(),
        }

    def test_generate(self) -> None:
        """Test the generate method of the Bayesian Network Classifier."""
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
        generated_bn = SemiParametricBayesianNetworkAugmentedNaiveBayes.generate(
            seed=SEED
        )
        assert isinstance(generated_bn, BaseBayesianNetworkClassifier)
        assert generated_bn.num_nodes() == len(generated_bn.feature_names_in_) + 1
        assert set(generated_bn.arcs()) == generated_arc_set


class TestSemiParametricNaiveBayes(BaseTestSemiParametricBayesianNetworkClassifier):
    bn_class = SemiParametricNaiveBayes
    model_filename = "spnbnc.pkl"
    str_representation = "Naive SemiParametric Bayesian Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
        )

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        return {
            TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
            "a": pbn.CKDEType(),
            "b": pbn.CKDEType(),
            "c": pbn.CKDEType(),
        }


class TestSemiParametricSelectiveNaiveBayes(
    BaseTestSemiParametricBayesianNetworkClassifier
):
    bn_class = SemiParametricSelectiveNaiveBayes
    model_filename = "spsnbnc.pkl"
    str_representation = "Selective Naive SemiParametric Bayesian Network Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        return {(TRUE_CLASS_LABEL, "b"), (TRUE_CLASS_LABEL, "c")}


class TestSemiParametricTreeAugmentedNaiveBayes(
    BaseTestSemiParametricBayesianNetworkClassifier
):
    bn_class = SemiParametricTreeAugmentedNaiveBayes
    model_filename = "sptanbnc.pkl"
    str_representation = (
        "Tree Augmented Naive SemiParametric Bayesian Network Classifier"
    )

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
            + [("c", "b"), ("a", "c")]
        )

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        return {
            TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
            "a": pbn.CKDEType(),
            "b": pbn.CKDEType(),
            "c": pbn.CKDEType(),
        }


class TestSemiParametricSuperParentOneDependenceEstimator(
    BaseTestSemiParametricBayesianNetworkClassifier
):
    bn_class = SemiParametricSuperParentOneDependenceEstimator
    model_filename = "spsodbnc.pkl"
    str_representation = (
        "Superparent-one-dependence SemiParametric Bayesian Network Classifier"
    )
    init_params = {"super_parent": SUPER_PARENT}

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.drop(columns=[TRUE_CLASS_LABEL]).columns
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes]
            + [(SUPER_PARENT, node) for node in nodes if node != SUPER_PARENT]
        )

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        return {
            TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
            "a": pbn.CKDEType(),
            "b": pbn.CKDEType(),
            "c": pbn.CKDEType(),
        }


class TestSemiParametricAveragedOneDependenceEstimator(
    BaseTestSemiParametricBayesianNetworkClassifier
):
    bn_class = SemiParametricAveragedOneDependenceEstimator
    model_filename = "spaodebnc.pkl"
    str_representation = (
        "Averaged one-dependence SemiParametric Bayesian Network Classifier"
    )
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


class TestSemiParametricKDependenceBayesian(
    BaseTestSemiParametricBayesianNetworkClassifier
):
    bn_class = SemiParametricKDependenceBayesian
    model_filename = "spkdbnc.pkl"
    str_representation = "k-Dependence SemiParametric Bayesian Network Classifier"
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


class TestSemiParametricMaxKAugmentedNaiveBayes(
    BaseTestSemiParametricBayesianNetworkClassifier
):
    bn_class = SemiParametricMaxKAugmentedNaiveBayes
    model_filename = "spmaxkbnc.pkl"
    str_representation = "Max-k SemiParametric Bayesian Network Classifier"
    init_params = {"max_indegree": 1}

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        nodes = data.columns.tolist()
        return set(
            [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
            + [("c", "a")]
        )

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        return {
            TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
            "a": pbn.CKDEType(),
            "b": pbn.CKDEType(),
            "c": pbn.CKDEType(),
        }

    def has_max_indegree_constraint(self) -> bool:
        return True


class TestSemiParametricBayesianMultinet(
    BaseTestSemiParametricBayesianNetworkClassifier
):
    bn_class = SemiParametricBayesianMultinet
    model_filename = "spbmc.pkl"
    str_representation = "SemiParametric Bayesian Multinet Classifier"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        # Multinet doesn't have fixed arc structure
        return {
            "class1": [("c", "a"), ("c", "b")],
            "class2": [("c", "b"), ("b", "a"), ("c", "a")],
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
                "c": pbn.LinearGaussianCPDType(),
            },
        }
