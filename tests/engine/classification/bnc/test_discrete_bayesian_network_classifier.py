# TODO: Add when discrete BN is implemented
# import numpy as np
# import pandas as pd
# import pytest
# from helpers.data import SEED, SUPER_PARENT, TRUE_CLASS_LABEL
# from .test_bayesian_network_classifier import DiscreteBaseTestBayesianNetworkClassifier


# class TestDiscreteBayesianNetworkClassifier(DiscreteBaseTestBayesianNetworkClassifier):
#     @pytest.fixture(scope="class")
#     def bn(self, data: pd.DataFrame) -> BayesianNetworkClassifier:
#         """Generates a Bayesian Network Classifier.

#         Args:
#             data (pd.DataFrame): The data.

#         Returns:
#             BayesianNetworkClassifier: The Bayesian Network Classifier.
#         """
#         bn = DiscreteBayesianNetworkClassifier(true_label=TRUE_CLASS_LABEL, seed=SEED)
#         bn.fit(data.drop(columns=[TRUE_CLASS_LABEL]), data[TRUE_CLASS_LABEL])
#         return bn

#     def test_initialization(
#         self, bn: BayesianNetworkClassifier, data: pd.DataFrame
#     ) -> None:
#         """Test the initialization of the Discrete Bayesian Network Classifier.

#         Args:
#             bn (BayesianNetwork): The Discrete Bayesian Network Classifier.
#             data (pd.DataFrame): The data.
#         """
#         super().test_initialization(bn, data)

#         nodes = data.columns.tolist()
#         arc_set = set(
#             [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
#         )
#         assert set(bn.arcs()) == arc_set

#     def test_str(self, bn: BayesianNetworkClassifier) -> None:
#         """Test the str method of the Bayesian Network Classifier.

#         Args:
#             bn (BayesianNetworkClassifier): The Bayesian Network Classifier.
#         """
#         assert (
#             str(bn)
#             == f"Discrete Bayesian Network Classifier with {bn.num_nodes()} nodes and {bn.num_arcs()} arcs"
#         )


# class TestDiscreteNaiveBayes(
#     DiscreteBaseTestBayesianNetworkClassifier
# ):

#     @pytest.fixture(scope="class")
#     def bn(self, data: pd.DataFrame) -> BayesianNetworkClassifier:
#         """Generates a Bayesian Network Classifier.

#         Args:
#             data (pd.DataFrame): The data.

#         Returns:
#             BayesianNetworkClassifier: The Bayesian Network Classifier.
#         """
#         bn = DiscreteNaiveBayes(
#             true_label=TRUE_CLASS_LABEL,
#             seed=SEED,
#         )
#         bn.fit(data.drop(columns=[TRUE_CLASS_LABEL]), data[TRUE_CLASS_LABEL])
#         return bn

#     def test_initialization(
#         self, bn: BayesianNetworkClassifier, data: pd.DataFrame
#     ) -> None:
#         """Test the initialization of the Bayesian Network.

#         Args:
#             bn (BayesianNetwork): The Bayesian Network.
#             data (pd.DataFrame): The data.
#         """
#         super().test_initialization(bn, data)

#         nodes = data.columns.tolist()
#         arc_set = set(
#             [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
#         )
#         assert set(bn.arcs()) == arc_set

#     def test_str(self, bn: BayesianNetworkClassifier) -> None:
#         """Test the str method of the Bayesian Network Classifier.

#         Args:
#             bn (BayesianNetworkClassifier): Bayesian Network Classifier.
#         """
#         assert (
#             str(bn)
#             == f"Naive Discrete Bayesian Network Classifier with {bn.num_nodes()} nodes and {bn.num_arcs()} arcs"
#         )


# class TestDiscreteSelectiveNaiveBayes(
#     DiscreteBaseTestBayesianNetworkClassifier
# ):

#     @pytest.fixture(scope="class")
#     def bn(self, data: pd.DataFrame) -> BayesianNetworkClassifier:
#         """Generates a Bayesian Network Classifier.

#         Args:
#             data (pd.DataFrame): The data.

#         Returns:
#             BayesianNetworkClassifier: The Bayesian Network Classifier.
#         """
#         bn = DiscreteSelectiveNaiveBayes(
#             true_label=TRUE_CLASS_LABEL,
#             seed=SEED,
#         )
#         bn.fit(data.drop(columns=[TRUE_CLASS_LABEL]), data[TRUE_CLASS_LABEL])
#         return bn

#     def test_initialization(
#         self, bn: BayesianNetworkClassifier, data: pd.DataFrame
#     ) -> None:
#         """Test the initialization of the Bayesian Network.

#         Args:
#             bn (BayesianNetwork): The Bayesian Network.
#             data (pd.DataFrame): The data.
#         """
#         super().test_initialization(bn, data)

#         nodes = data.columns.tolist()
#         arc_set = set(
#             [(TRUE_CLASS_LABEL, "b"), (TRUE_CLASS_LABEL, "c"), (TRUE_CLASS_LABEL, "a")]
#         )
#         assert set(bn.arcs()) == arc_set

#     def test_str(self, bn: BayesianNetworkClassifier) -> None:
#         """Test the str method of the Bayesian Network Classifier.

#         Args:
#             bn (BayesianNetworkClassifier): Bayesian Network Classifier.
#         """
#         assert (
#             str(bn)
#             == f"Selective Naive Discrete Bayesian Network Classifier with {bn.num_nodes()} nodes and {bn.num_arcs()} arcs"
#         )


# class TestDiscreteTreeAugmentedNaiveBayes(
#     DiscreteBaseTestBayesianNetworkClassifier
# ):
#     @pytest.fixture(scope="class")
#     def bn(self, data: pd.DataFrame) -> BayesianNetworkClassifier:
#         """Generates a Bayesian Network Classifier.

#         Args:
#             data (pd.DataFrame): The data.

#         Returns:
#             BayesianNetworkClassifier: The Bayesian Network Classifier.
#         """
#         bn = DiscreteTreeAugmentedNaiveBayes(
#             true_label=TRUE_CLASS_LABEL, seed=SEED
#         )
#         bn.fit(data.drop(columns=[TRUE_CLASS_LABEL]), data[TRUE_CLASS_LABEL])
#         return bn

#     def test_initialization(
#         self, bn: BayesianNetworkClassifier, data: pd.DataFrame
#     ) -> None:
#         """Test the initialization of the Bayesian Network.

#         Args:
#             bn (BayesianNetwork): The Bayesian Network.
#             data (pd.DataFrame): The data.
#         """
#         super().test_initialization(bn, data)
#         nodes = data.columns.tolist()
#         arc_set = set(
#             [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
#             + [
#                 ("b", "c"),
#                 ("c", "a"),
#             ]
#         )

#         assert set(bn.arcs()) == arc_set

#     def test_str(self, bn: BayesianNetworkClassifier) -> None:
#         """Test the str method of the Bayesian Network Classifier.

#         Args:
#             bn (BayesianNetworkClassifier): Bayesian Network Classifier.
#         """
#         assert (
#             str(bn)
#             == f"Tree Augmented Naive Discrete Bayesian Network Classifier with {bn.num_nodes()} nodes and {bn.num_arcs()} arcs"
#         )


# class TestDiscreteSuperParentOneDependenceEstimator(
#     DiscreteBaseTestBayesianNetworkClassifier
# ):

#     @pytest.fixture(scope="class")
#     def bn(self, data: pd.DataFrame) -> BayesianNetworkClassifier:
#         """Generates a Bayesian Network Classifier.

#         Args:
#             data (pd.DataFrame): The data.

#         Returns:
#             BayesianNetworkClassifier: The Bayesian Network Classifier.
#         """
#         bn = DiscreteSuperParentOneDependenceEstimator(
#             true_label=TRUE_CLASS_LABEL,
#             seed=SEED,
#             super_parent=SUPER_PARENT,
#         )
#         bn.fit(data.drop(columns=[TRUE_CLASS_LABEL]), data[TRUE_CLASS_LABEL])
#         return bn

#     def test_initialization(
#         self, bn: BayesianNetworkClassifier, data: pd.DataFrame
#     ) -> None:
#         """Test the initialization of the Bayesian Network.

#         Args:
#             bn (BayesianNetwork): The Bayesian Network.
#             data (pd.DataFrame): The data.
#         """
#         super().test_initialization(bn, data)

#         nodes = data.drop(columns=[TRUE_CLASS_LABEL]).columns
#         arc_set = set(
#             [(TRUE_CLASS_LABEL, node) for node in nodes]
#             + [(SUPER_PARENT, node) for node in nodes if node != SUPER_PARENT]
#         )
#         assert set(bn.arcs()) == arc_set

#     def test_str(self, bn: BayesianNetworkClassifier) -> None:
#         """Test the str method of the Bayesian Network Classifier.

#         Args:
#             bn (BayesianNetworkClassifier): Bayesian Network Classifier.
#         """
#         assert (
#             str(bn)
#             == f"Superparent-one-dependence Discrete Bayesian Network Classifier with {bn.num_nodes()} nodes and {bn.num_arcs()} arcs"
#         )


# # TODO: check max_indegree works with better example
# class TestDiscreteMaxKAugmentedNaiveBayes(
#     DiscreteBaseTestBayesianNetworkClassifier
# ):

#     @pytest.fixture(scope="class")
#     def bn(self, data: pd.DataFrame) -> BayesianNetworkClassifier:
#         """Generates a Bayesian Network Classifier.

#         Args:
#             data (pd.DataFrame): The data.

#         Returns:
#             BayesianNetworkClassifier: The Bayesian Network Classifier.
#         """
#         bn = DiscreteMaxKAugmentedNaiveBayes(
#             true_label=TRUE_CLASS_LABEL,
#             seed=SEED,
#             max_indegree=1,
#         )
#         bn.fit(data.drop(columns=[TRUE_CLASS_LABEL]), data[TRUE_CLASS_LABEL])
#         return bn

#     def test_initialization(
#         self, bn: BayesianNetworkClassifier, data: pd.DataFrame
#     ) -> None:
#         """Test the initialization of the Bayesian Network.

#         Args:
#             bn (BayesianNetwork): The Bayesian Network.
#             data (pd.DataFrame): The data.
#         """
#         super().test_initialization(bn, data)

#         nodes = data.columns.tolist()
#         arc_set = set(
#             [(TRUE_CLASS_LABEL, node) for node in nodes if node != TRUE_CLASS_LABEL]
#         )
#         assert set(bn.arcs()) == arc_set

#     def test_str(self, bn: BayesianNetworkClassifier) -> None:
#         """Test the str method of the Bayesian Network Classifier.

#         Args:
#             bn (BayesianNetworkClassifier): Bayesian Network Classifier.
#         """
#         assert (
#             str(bn)
#             == f"Max-k Discrete Bayesian Network Classifier with {bn.num_nodes()} nodes and {bn.num_arcs()} arcs"
#         )


# # RFE: finish pytesting
# class TestDiscreteBayesianMultinet(DiscreteBaseTestBayesianNetworkClassifier):
#     @pytest.fixture(scope="class")
#     def bn(self, data: pd.DataFrame) -> BayesianNetworkClassifier:
#         """Generates a Bayesian Network Classifier.

#         Args:
#             data (pd.DataFrame): The data.

#         Returns:
#             BayesianNetworkClassifier: The Bayesian Network Classifier.
#         """
#         bn = DiscreteBayesianMultinet(
#             true_label=TRUE_CLASS_LABEL,
#             seed=SEED,
#             max_indegree=1,
#         )
#         bn.fit(data.drop(columns=[TRUE_CLASS_LABEL]), data[TRUE_CLASS_LABEL])
#         return bn

#     def test_str(self, bn: BayesianNetworkClassifier) -> None:
#         """Test the str method of the Bayesian Network Classifier.

#         Args:
#             bn (BayesianNetworkClassifier): Bayesian Network Classifier.
#         """
#         assert str(bn) == f"Discrete Bayesian Multinet Classifier"

#     def test_feature_logl(
#         self, bn: BayesianNetworkClassifier, data: pd.DataFrame
#     ) -> None:
#         """Test the feature_logl method of the Bayesian Network.

#         Args:
#             bn (BayesianNetwork): The Bayesian Network.
#             data (pd.DataFrame): The data.
#         """

#         feature_logl_df = bn.feature_logl(data)
#         pll_columns = [n + "_pll" for n in data.columns if n != TRUE_CLASS_LABEL]
#         assert isinstance(feature_logl_df, pd.DataFrame)
#         assert feature_logl_df.shape[0] == data.shape[0]
#         assert all([c in feature_logl_df.columns for c in pll_columns])
#         np.testing.assert_allclose(
#             feature_logl_df["log_likelihood"],
#             feature_logl_df[pll_columns].sum(axis=1)
#             + np.log(bn.weights_.to_numpy()).sum(),
#             rtol=1e-5,
#             atol=1e-8,
#         )
#         np.testing.assert_allclose(
#             feature_logl_df["log_likelihood"],
#             bn.logl(data),
#             rtol=1e-5,
#             atol=1e-8,
#         )
