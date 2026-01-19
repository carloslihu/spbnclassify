import numpy as np
import pandas as pd
import pybnesian as pbn
from engine.classification.bn import BaseTestBayesianNetwork
from helpers.data import (
    TRUE_CLASS_LABEL,
    generate_discrete_data_classification,
    generate_non_normal_data_classification,
    generate_normal_data_classification,
)

# TODO: Remove rutile_ai dependency
from rutile_ai.engine.classification.spbnclassify.src.spbnclassify.bnc import (
    BaseBayesianNetworkClassifier,
)


class BaseTestBayesianNetworkClassifier(BaseTestBayesianNetwork):
    """Base test class for Bayesian Network Classifiers."""

    # Configuration mapping for data types and expected network types
    DATA_GENERATORS = {
        "discrete": generate_discrete_data_classification,
        "gaussian": generate_normal_data_classification,
        "non_gaussian": generate_non_normal_data_classification,
    }

    def test_predict_proba(
        self, bn: BaseBayesianNetworkClassifier, data: pd.DataFrame
    ) -> None:
        """Test the predict_proba method of the Bayesian Network Classifier."""
        super().test_predict_proba(bn, data)
        # Additional check for classification: probabilities should sum to 1
        np.testing.assert_allclose(
            self._last_predict_proba.sum(axis=1), 1.0, rtol=1e-5, atol=1e-8
        )

    def test_predict(
        self, bn: BaseBayesianNetworkClassifier, data: pd.DataFrame
    ) -> None:
        """Test the predict method of the Bayesian Network Classifier."""
        predicted_classes = bn.predict(data)
        # Check that all predicted classes are valid
        assert set(predicted_classes).issubset(set(bn.classes_))


class BaseTestDiscreteBayesianNetworkClassifier(BaseTestBayesianNetworkClassifier):
    """Test class for Discrete Bayesian Network Classifiers."""

    data_type = "discrete"
    expected_bn_type_key = "discrete"


class BaseTestGaussianBayesianNetworkClassifier(BaseTestBayesianNetworkClassifier):
    """Test class for Gaussian Bayesian Network Classifiers."""

    data_type = "gaussian"
    expected_bn_type_key = "gaussian"

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        return {
            TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
            "a": pbn.LinearGaussianCPDType(),
            "b": pbn.LinearGaussianCPDType(),
            "c": pbn.LinearGaussianCPDType(),
        }


class BaseTestSemiParametricBayesianNetworkClassifier(
    BaseTestBayesianNetworkClassifier
):
    """Test class for SemiParametric Bayesian Network Classifiers."""

    data_type = "non_gaussian"
    expected_bn_type_key = "semiparametric"


class BaseTestKDEBayesianNetworkClassifier(BaseTestBayesianNetworkClassifier):
    """Test class for KDE Bayesian Network Classifiers."""

    data_type = "non_gaussian"
    expected_bn_type_key = "kde"

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        return {
            TRUE_CLASS_LABEL: pbn.DiscreteFactorType(),
            "a": pbn.CKDEType(),
            "b": pbn.CKDEType(),
            "c": pbn.CKDEType(),
        }
