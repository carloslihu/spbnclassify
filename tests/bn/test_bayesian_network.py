import json
from abc import abstractmethod
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pybnesian as pbn
import pytest
from helpers.data import (
    BN_SAVE_FOLDER_PATH,
    DATA_SIZE,
    SAMPLE_SIZE,
    SEED,
    generate_discrete_data,
    generate_non_normal_data,
    generate_normal_data,
)
from scipy.special import logsumexp
from scipy.stats import norm
from src.bn import (
    BayesianNetwork,
    DiscreteBayesianNetwork,
    GaussianBayesianNetwork,
    KDEBayesianNetwork,
    SemiParametricBayesianNetwork,
)


def _conditional_gaussian(joint_mean, joint_cov, query_nodes, evidence):
    """
    Compute the conditional mean and covariance for a set of query nodes given evidence.
    """
    evidence_nodes = list(evidence)
    mu_q = joint_mean[query_nodes].to_numpy(dtype=float).ravel()
    mu_e = joint_mean[evidence_nodes].to_numpy(dtype=float).ravel()
    cov_qq = joint_cov.loc[query_nodes, query_nodes].to_numpy(dtype=float)
    cov_qe = joint_cov.loc[query_nodes, evidence_nodes].to_numpy(dtype=float)
    cov_eq = joint_cov.loc[evidence_nodes, query_nodes].to_numpy(dtype=float)
    cov_ee = joint_cov.loc[evidence_nodes, evidence_nodes].to_numpy(dtype=float)

    e = np.array([evidence[node] for node in evidence_nodes], dtype=float)
    gain = cov_qe @ np.linalg.inv(cov_ee)
    cond_mean = mu_q + gain @ (e - mu_e)
    cond_cov = cov_qq - gain @ cov_eq
    return cond_mean, cond_cov


class BaseTestBayesianNetwork:
    """
    BaseTestBayesianNetwork provides a reusable base class for testing Bayesian Network classifiers.
    It defines common test methods and fixtures to ensure consistent and comprehensive testing of
    different Bayesian Network implementations. Subclasses should override configuration attributes
    and abstract methods to specify the data type, expected network structure, and node types.
    Attributes:
        data_type (str): The type of data to generate for testing (e.g., "discrete", "gaussian").
        expected_bn_type_key (str): Key to select the expected Bayesian Network type for assertions.
        bn_class (type): The Bayesian Network classifier class under test.
        model_filename (str): Filename for saving/loading the trained model.
        str_representation (str): Expected string representation of the model.
        init_params (dict): Initialization parameters for the classifier.
        DATA_GENERATORS (dict): Mapping of data types to data generator functions.
        EXPECTED_BN_TYPES (dict): Mapping of type keys to expected Bayesian Network type instances.
    Fixtures:
        data: Generates test data using the appropriate generator based on data_type.
        bn: Loads or creates a Bayesian Network model for testing.
    Abstract Methods:
        get_expected_arcs(data): Returns the expected arcs (edges) for the Bayesian Network.
        get_expected_node_types(data): Returns the expected node types for the Bayesian Network.
    Methods:
        has_max_indegree_constraint(): Indicates if a max indegree constraint should be tested.
        create_bn(data): Instantiates and fits a Bayesian Network classifier.
        create_and_save_bn(data, force_recreate): Creates and saves a Bayesian Network model.
        test_initialization(bn, data): Tests initialization, structure, and node types.
        test_fit(bn, data): Tests the fit method and model consistency.
        test_logl(bn, data): Tests the log-likelihood computation.
        test_slogl(bn, data): Tests the sum log-likelihood computation.
        test_predict_proba(bn, data): Tests the probability prediction method.
        test_sample(bn, data): Tests the sampling method for data generation.
        test_save_load(tmp_path, bn): Tests saving and loading of the model.
        test_generate(): Placeholder for testing the generate method.
    Note:
        Some advanced tests (e.g., feature_logl, anomaly_score, explainability) are included as
        commented-out methods for future implementation.
    """

    # NOTE: Override these in subclasses
    data_type = None
    expected_bn_type_key = None
    bn_class = None
    model_filename = None
    str_representation = None
    init_params = {}

    # Configuration mapping for data types and expected network types
    DATA_GENERATORS = {
        "discrete": generate_discrete_data,
        "gaussian": generate_normal_data,
        "non_gaussian": generate_non_normal_data,
    }
    EXPECTED_BN_TYPES = {
        "discrete": pbn.DiscreteBNType(),
        "gaussian": pbn.CLGNetworkType(),
        "semiparametric": pbn.SemiparametricBNType(),
        "kde": pbn.SemiparametricBNType(),
    }

    @pytest.fixture(scope="class")
    def data(self) -> pd.DataFrame:
        """Automatically use the correct data generator based on data_type."""
        if self.data_type not in self.DATA_GENERATORS:
            raise ValueError(f"Unknown data type: {self.data_type}")
        return self.DATA_GENERATORS[self.data_type](DATA_SIZE, seed=SEED)

    @abstractmethod
    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        """
        Returns the expected arcs (edges) for a Bayesian network based on the provided DataFrame.
        Args:
            data (pd.DataFrame): The input data used to determine the expected arcs.
        Returns:
            set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
                Either a set of tuples representing directed arcs (from, to) between variables,
                or a dictionary mapping strings to lists of such tuples, depending on the implementation.
        Raises:
            NotImplementedError: If the method is not implemented.
        """

    @abstractmethod
    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        """
        Determine the expected node types for each column in the given DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data to analyze.

        Returns:
            dict: A dictionary mapping column names to their expected node types.
        """

    def has_max_indegree_constraint(self) -> bool:
        """
        Checks whether the Bayesian network has a maximum indegree constraint.
        Returns:
            bool: False if there is no maximum indegree constraint, True otherwise.
        """

        return False

    def create_bn(self, data: pd.DataFrame) -> BayesianNetwork:
        """Create a Bayesian Network."""
        # Ensure bn_class is set before calling
        if self.bn_class is None:
            raise NotImplementedError(
                "bn_class must be set to a valid classifier class before calling create_bn."
            )
        bn = self.bn_class(seed=SEED)
        if bn.true_label in data.columns:
            X = data.drop(columns=[bn.true_label])
            y = data[bn.true_label]
        else:
            X = data
            y = None
        bn.fit(X, y)
        return bn

    def create_and_save_bn(
        self, data: pd.DataFrame, force_recreate: bool = False
    ) -> None:
        """Create and save a Bayesian Network for testing.

        Args:
            data: The training data
            force_recreate: If True, recreate even if file exists
        """
        if not self.bn_class or not self.model_filename:
            raise NotImplementedError("bn_class and model_filename must be set")

        model_path = BN_SAVE_FOLDER_PATH / self.model_filename

        # Only create if doesn't exist or force_recreate is True
        if not model_path.exists() or force_recreate:
            BN_SAVE_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
            print(f"Creating and saving BN model: {self.model_filename}")
            bn = self.create_bn(data)
            bn.save(model_path)
            print(f"Model saved to: {model_path}")
        else:
            print(f"Model already exists: {model_path}")

    @pytest.fixture(scope="class")
    def bn(self, data: pd.DataFrame) -> BayesianNetwork:
        """Generic BN fixture that uses class configuration."""
        if not self.bn_class or not self.model_filename:
            raise NotImplementedError("bn_class and model_filename must be set")

        # Load the saved model to ensure it works correctly
        model_path = BN_SAVE_FOLDER_PATH / self.model_filename
        if not model_path.exists():
            self.create_and_save_bn(data, force_recreate=False)
        bn = BayesianNetwork.load(model_path)
        return bn

    def test_initialization(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
        """Generic initialization test that uses get_expected_arcs and get_expected_node_types."""
        assert set(bn.nodes()) == set(data.columns)

        if self.expected_bn_type_key:
            expected_type = self.EXPECTED_BN_TYPES[self.expected_bn_type_key]
            assert bn.bn_type == expected_type
        expected_arcs = self.get_expected_arcs(data)
        arcs = bn.arcs()
        if expected_arcs is not None:
            if isinstance(arcs, list):
                # For multinet structures, arcs is a list of tuples
                assert set(arcs) == set(expected_arcs)
            elif isinstance(arcs, dict):
                # For single network structures, arcs is a set of tuples
                for key, expected_arc_list in expected_arcs.items():
                    arc_list = arcs[key]
                    assert set(arc_list) == set(expected_arc_list)

        expected_node_types = self.get_expected_node_types(data)
        node_types = bn.node_types()
        if expected_node_types is not None:
            assert node_types == expected_node_types

        # Test max_indegree constraint if applicable
        if self.has_max_indegree_constraint():
            arc_set = set(bn.arcs())
            indegree_count = Counter([dst for _, dst in arc_set])
            max_indegree = (
                max(indegree_count.values()) - 1
            )  # Don't count class variable
            assert max_indegree <= bn.max_indegree

        # TODO: Check that parameters are the same for Discrete and CLG
        # Check that the bn and its pyagrum graphic are consistent in terms of nodes and arcs
        graphic_arcs_named = {
            (bn.graphic.variable(source).name(), bn.graphic.variable(target).name())
            for source, target in bn.graphic.arcs()
        }
        assert set(bn.nodes()) == set(bn.graphic.names())
        assert set(bn.arcs()) == graphic_arcs_named

    # def test_str(self, bn: BaseBayesianNetworkClassifier) -> None:
    #     """Generic string test that uses str_representation."""
    #     if not self.str_representation:
    #         raise NotImplementedError("str_representation must be set")

    #     if "Multinet" in self.str_representation:
    #         assert str(bn) == self.str_representation
    #     else:
    #         expected = f"{self.str_representation} with {bn.num_nodes()} nodes and {bn.num_arcs()} arcs"
    #         assert str(bn) == expected

    def test_fit(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
        """Test the fit method of the Bayesian Network."""
        if bn.true_label in data.columns:
            X = data.drop(columns=[bn.true_label])
            y = data[bn.true_label]
        else:
            X = data
            y = None
        new_bn = type(bn)(seed=SEED)
        new_bn.fit(X, y)

        assert new_bn.fitted()
        assert set(new_bn.nodes()) == set(bn.nodes())
        assert set(new_bn.arcs()) == set(bn.arcs())
        assert new_bn.node_types() == bn.node_types()

    def test_logl(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
        """Test the logl method of the Bayesian Network."""
        logl = bn.logl(data)
        assert isinstance(logl, np.ndarray)
        assert logl.shape[0] == data.shape[0]
        assert np.all(logl <= 0)

    def test_slogl(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
        """Test the slogl method of the Bayesian Network."""
        slogl = bn.slogl(data)
        assert isinstance(slogl, float)
        assert np.all(slogl <= 0)

    def test_predict_proba(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
        """Test the predict_proba method of the Bayesian Network."""
        proba = bn.predict_proba(data)
        assert isinstance(proba, np.ndarray)
        assert proba.shape[0] == data.shape[0]
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        # NOTE: Store it as an attribute for children checks
        self._last_predict_proba = proba

    def test_sample(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
        """Test the sample method of the Bayesian Network."""
        sample = bn.sample(SAMPLE_SIZE)
        assert isinstance(sample, pd.DataFrame)
        assert sample.shape[0] == SAMPLE_SIZE
        assert sample.shape[1] == data.shape[1]
        # Assert that the float sampled data is within the range of the original data

        # RFE: Check that the sample is within the original distribution
        # float_columns = data.select_dtypes(include=["float64"]).columns
        # assert np.all(sample[float_columns] >= data[float_columns].min())
        # assert np.all(sample[float_columns] <= data[float_columns].max())

        # Assert that the categorical sampled data is within the range of the original data
        categorical_columns = data.select_dtypes(include=["category"]).columns
        for col in categorical_columns:
            original_categories = set(data[col].cat.categories)
            sampled_categories = set(sample[col].unique())
            assert sampled_categories.issubset(
                original_categories
            ), f"Sampled categories in column {col} are not valid."

    def test_save_load(self, tmp_path: Path, bn: BayesianNetwork) -> None:
        """Test the save and load methods of the Bayesian Network."""
        model_file = tmp_path / "model.pkl"
        arc_set = set(bn.arcs())
        node_set = set(bn.nodes())
        bn.save(model_file)
        loaded_bn = type(bn).load(model_file)

        assert set(loaded_bn.nodes()) == node_set
        assert set(loaded_bn.arcs()) == arc_set
        assert loaded_bn.node_types() == bn.node_types()

        graph_file = Path(str(model_file).replace(".pkl", ".pdf"))
        bn.save(graph_file)
        assert graph_file.exists()

    def _infer_result(self, bn: BayesianNetwork, evidence: dict) -> dict:
        """Compute the inference result shared by infer-related tests."""
        json_file_path = BN_SAVE_FOLDER_PATH / self.model_filename.replace(
            ".pkl", "_infer_result.json"
        )
        pdf_file_path = BN_SAVE_FOLDER_PATH / self.model_filename.replace(
            ".pkl", "_infer_result.pdf"
        )

        # Clean up any existing files before the test
        json_file_path.unlink(missing_ok=True)
        pdf_file_path.unlink(missing_ok=True)

        infer_dict = bn.infer(
            evidence=evidence,
            json_file_path=json_file_path,
            pdf_file_path=pdf_file_path,
        )
        assert isinstance(infer_dict, dict)
        assert json_file_path.exists()
        assert pdf_file_path.exists()

        return infer_dict

    def test_infer(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
        """Test the infer method."""
        evidence = {"b": 1}
        infer_dict = self._infer_result(bn, evidence)
        assert isinstance(infer_dict, dict)

    def _posterior_probability(self, bn: BayesianNetwork, data: pd.DataFrame) -> float:
        """Compute the posterior probability shared by posterior-related tests."""
        point = data.iloc[0]
        query_node = "a"
        evidence_b = {"b": point["b"]}

        # Compute P(a | b)
        prob_a_given_b = bn.posterior(
            query_node=query_node, evidence=evidence_b, point=point
        )

        assert np.isfinite(prob_a_given_b)
        return prob_a_given_b

    def test_posterior(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
        """Test the posterior method."""
        prob_a_given_b = self._posterior_probability(bn, data)
        assert np.isfinite(prob_a_given_b)
        assert prob_a_given_b >= 0

    # TODO: Add later
    # def test_feature_logl(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
    #     """Test the feature_logl method of the Bayesian Network.

    #     Args:
    #         bn (BayesianNetwork): The Bayesian Network.
    #         data (pd.DataFrame): The data.
    #     """
    #     feature_logl_df = bn.feature_logl(data)
    #     pll_columns = [n + "_pll" for n in data.columns]
    #     assert isinstance(feature_logl_df, pd.DataFrame)
    #     assert feature_logl_df.shape[0] == data.shape[0]
    #     assert all([c in feature_logl_df.columns for c in pll_columns])
    #     np.testing.assert_allclose(
    #         feature_logl_df["log_likelihood"],
    #         feature_logl_df.loc[:, pll_columns].sum(axis=1),
    #         rtol=1e-5,
    #         atol=1e-8,
    #     )
    #     np.testing.assert_allclose(
    #         feature_logl_df["log_likelihood"],
    #         bn.logl(data),
    #         rtol=1e-5,
    #         atol=1e-8,
    #     )

    # def test_anomaly_score(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
    #     """Test the anomaly_score method of the Bayesian Network.

    #     Args:
    #         bn (BayesianNetwork): The Bayesian Network.
    #         data (pd.DataFrame): The data.
    #     """
    #     anomaly_score = bn.anomaly_score(data)
    #     assert isinstance(anomaly_score, np.ndarray)
    #     assert anomaly_score.shape[0] == data.shape[0]
    #     assert np.all(anomaly_score >= 0)

    # def test_feature_anomaly_score(
    #     self, bn: BayesianNetwork, data: pd.DataFrame
    # ) -> None:
    #     """Test the feature_anomaly_score method of the Bayesian Network.

    #     Args:
    #         bn (BayesianNetwork): The Bayesian Network.
    #         data (pd.DataFrame): The data.
    #     """
    #     feature_anomaly_score_df = bn.feature_anomaly_score(data)
    #     score_columns = [n + "_score" for n in data.columns]
    #     assert all([c in feature_anomaly_score_df.columns for c in score_columns])
    #     assert isinstance(feature_anomaly_score_df, pd.DataFrame)
    #     assert feature_anomaly_score_df.shape[0] == data.shape[0]
    #     assert np.all(
    #         feature_anomaly_score_df["anomaly_score"]
    #         == feature_anomaly_score_df[score_columns].sum(axis=1)
    #     )
    #     np.testing.assert_allclose(
    #         feature_anomaly_score_df["anomaly_score"],
    #         bn.anomaly_score(data),
    #         rtol=1e-5,
    #         atol=1e-8,
    #     )

    # TODO: test_explain and test_show


class TestDiscreteBayesianNetwork(BaseTestBayesianNetwork):
    bn_class = DiscreteBayesianNetwork
    model_filename = "d-bn.pkl"
    data_type = "discrete"
    expected_bn_type_key = "discrete"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        expected_arcs = {("c", "a"), ("c", "b"), ("d", "c"), ("b", "a")}
        return expected_arcs

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        expected_node_types = {
            "a": pbn.DiscreteFactorType(),
            "b": pbn.DiscreteFactorType(),
            "c": pbn.DiscreteFactorType(),
            "d": pbn.DiscreteFactorType(),
        }
        return expected_node_types

    def test_infer(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
        """Test the infer method for discrete Bayesian networks."""
        evidence = {"b": 1}
        infer_dict = self._infer_result(bn, evidence)
        assert isinstance(infer_dict, dict)
        parameter_dict = infer_dict["parameters"]

        # Check that the parameters are valid probability distributions
        for node in bn.nodes():
            assert node in parameter_dict.keys()
            for value, prob in parameter_dict[node].items():
                assert isinstance(value, (int, float, str))
                assert isinstance(prob, float)
                assert 0 <= prob <= 1
            # Check that probabilities sum to 1
            total_prob = sum(parameter_dict[node].values())
            np.testing.assert_allclose(total_prob, 1.0, rtol=1e-5, atol=1e-8)

    def test_posterior(self, bn: BayesianNetwork, data: pd.DataFrame) -> None:
        """Test the posterior method."""
        prob_a_given_b = self._posterior_probability(bn, data)
        assert np.isfinite(prob_a_given_b)
        assert prob_a_given_b >= 0

        # Additional checks for discrete Bayesian networks
        assert prob_a_given_b <= 1


class TestGaussianBayesianNetwork(BaseTestBayesianNetwork):
    bn_class = GaussianBayesianNetwork
    model_filename = "g-bn.pkl"
    data_type = "gaussian"
    expected_bn_type_key = "gaussian"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        expected_arcs = {
            ("a", "c"),
            ("b", "c"),
            ("d", "c"),
            ("b", "a"),
            ("a", "d"),
            ("b", "d"),
        }
        return expected_arcs

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        expected_node_types = {
            "a": pbn.LinearGaussianCPDType(),
            "b": pbn.LinearGaussianCPDType(),
            "c": pbn.LinearGaussianCPDType(),
            "d": pbn.LinearGaussianCPDType(),
        }
        return expected_node_types

    def test_infer(self, bn: GaussianBayesianNetwork, data: pd.DataFrame) -> None:
        query_nodes = bn.nodes()
        evidence = {"b": data.iloc[0]["b"]}
        infer_dict = self._infer_result(bn, evidence)

        # Compute expected conditional mean and covariance using the joint Gaussian parameters
        expected_mean, expected_cov = _conditional_gaussian(
            bn.joint_gaussian_["mean"],
            bn.joint_gaussian_["cov"],
            query_nodes,
            evidence,
        )
        # Check that the inferred parameters match the expected conditional mean and covariance
        for i, node in enumerate(query_nodes):
            np.testing.assert_allclose(
                infer_dict["parameters"][node]["mean"],
                expected_mean[i],
                rtol=1e-5,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                infer_dict["parameters"][node]["std"],
                np.sqrt(expected_cov[i, i]),
                rtol=1e-5,
                atol=1e-8,
            )

    def test_posterior(self, bn: GaussianBayesianNetwork, data: pd.DataFrame) -> None:
        point = data.iloc[0]
        evidence = {"b": point["b"]}
        infer_dict = bn.infer(evidence=evidence)

        mu = infer_dict["parameters"]["a"]["mean"]
        std = infer_dict["parameters"]["a"]["std"]

        # Compute the expected posterior probability using the Gaussian PDF formula
        expected = np.exp(-0.5 * ((point["a"] - mu) / std) ** 2) / (
            np.sqrt(2 * np.pi) * std
        )
        # Compute the actual posterior probability using the bn.posterior method
        actual = bn.posterior("a", evidence=evidence, point=point)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


class TestKDEBayesianNetwork(BaseTestBayesianNetwork):
    bn_class = KDEBayesianNetwork
    model_filename = "kde-bn.pkl"
    data_type = "non_gaussian"
    expected_bn_type_key = "kde"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        expected_arcs = {("a", "c"), ("a", "d"), ("b", "d"), ("b", "c")}
        return expected_arcs

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        expected_node_types = {
            "a": pbn.CKDEType(),
            "b": pbn.CKDEType(),
            "c": pbn.CKDEType(),
            "d": pbn.CKDEType(),
        }
        return expected_node_types

    def test_infer(self, bn: KDEBayesianNetwork, data: pd.DataFrame) -> None:
        evidence = {"b": data.iloc[0]["b"]}
        infer_dict = bn.infer(evidence=evidence, n_samples=250, seed=SEED)

        assignments = infer_dict["parameters"]["assignments"]
        weights = infer_dict["parameters"]["weights"]

        assert set(assignments.columns) == set(bn.nodes())
        assert np.all(assignments["b"] == evidence["b"])

        assert np.all(weights >= 0)
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12)

        topo_order = list(bn.graph().topological_sort())
        log_weights = np.zeros(len(assignments), dtype=float)

        for node in topo_order:
            if node not in evidence:
                continue

            cpd = bn.cpd(node)
            parents = cpd.evidence()
            point_df = (
                assignments[parents].copy()
                if parents
                else pd.DataFrame(index=assignments.index)
            )
            point_df.insert(0, node, evidence[node])
            log_weights += np.asarray(cpd.logl(point_df), dtype=float)

        expected_weights = np.exp(log_weights - logsumexp(log_weights))
        np.testing.assert_allclose(weights, expected_weights, rtol=1e-10, atol=1e-12)

    def test_posterior(self, bn: KDEBayesianNetwork, data: pd.DataFrame) -> None:
        point = pd.Series({"a": 1.5})
        samples = np.array([0.0, 1.0, 2.0, 4.0])
        weights = np.array([0.1, 0.2, 0.6, 0.1])

        lw = {
            "evidence": {},
            "method": "lw",
            "parameters": {
                "assignments": pd.DataFrame({"a": samples}),
                "weights": weights,
                "posterior_std": {},
            },
        }

        # Weighted Silverman's rule with the effective sample size
        weighted_mean = np.sum(weights * samples)
        weighted_std = np.sqrt(np.sum(weights * (samples - weighted_mean) ** 2))
        n_eff = 1.0 / np.sum(weights**2)
        bandwidth = 1.06 * weighted_std * (n_eff ** (-1.0 / 5.0))
        kernel_values = np.exp(-0.5 * ((point["a"] - samples) / bandwidth) ** 2) / (
            np.sqrt(2.0 * np.pi) * bandwidth
        )
        expected = np.sum(weights * kernel_values)

        actual = bn.posterior(
            query_node="a",
            evidence={},
            point=point,
            infer_dict=lw,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


class TestSemiParametricBayesianNetwork(BaseTestBayesianNetwork):
    bn_class = SemiParametricBayesianNetwork
    model_filename = "sp-bn.pkl"
    data_type = "non_gaussian"
    expected_bn_type_key = "semiparametric"

    def get_expected_arcs(
        self, data: pd.DataFrame
    ) -> set[tuple[str, str]] | dict[str, list[tuple[str, str]]]:
        expected_arcs = {("a", "d"), ("c", "d"), ("a", "c"), ("b", "d"), ("c", "b")}
        return expected_arcs

    def get_expected_node_types(self, data: pd.DataFrame) -> dict:
        expected_node_types = {
            "a": pbn.CKDEType(),
            "b": pbn.CKDEType(),
            "c": pbn.CKDEType(),
            "d": pbn.LinearGaussianCPDType(),
        }
        return expected_node_types

    def test_infer(self, bn: SemiParametricBayesianNetwork, data: pd.DataFrame) -> None:
        evidence = {"b": data.iloc[0]["b"]}
        infer_dict = bn.infer(evidence=evidence, n_samples=250, seed=SEED)

        assignments = infer_dict["parameters"]["assignments"]
        weights = infer_dict["parameters"]["weights"]

        assert set(assignments.columns) == set(bn.nodes())
        assert np.all(assignments["b"] == evidence["b"])

        assert np.all(weights >= 0)
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12)

        topo_order = list(bn.graph().topological_sort())
        log_weights = np.zeros(len(assignments), dtype=float)

        for node in topo_order:
            if node not in evidence:
                continue

            cpd = bn.cpd(node)
            parents = cpd.evidence()
            point_df = (
                assignments[parents].copy()
                if parents
                else pd.DataFrame(index=assignments.index)
            )
            point_df.insert(0, node, evidence[node])
            log_weights += np.asarray(cpd.logl(point_df), dtype=float)

        expected_weights = np.exp(log_weights - logsumexp(log_weights))
        np.testing.assert_allclose(weights, expected_weights, rtol=1e-10, atol=1e-12)

    def test_posterior(
        self, bn: SemiParametricBayesianNetwork, data: pd.DataFrame
    ) -> None:
        # NOTE: KDE query node case
        point = pd.Series({"a": 1.5, "d": 2.0})
        samples = np.array([0.0, 1.0, 2.0, 4.0])
        weights = np.array([0.1, 0.2, 0.6, 0.1])

        lw = {
            "evidence": {},
            "method": "lw",
            "parameters": {
                "assignments": pd.DataFrame({"a": samples, "d": samples}),
                "weights": weights,
                "posterior_std": {},
            },
        }

        # Weighted Silverman's rule with the effective sample size
        weighted_mean = np.sum(weights * samples)
        weighted_std = np.sqrt(np.sum(weights * (samples - weighted_mean) ** 2))
        n_eff = 1.0 / np.sum(weights**2)
        bandwidth = 1.06 * weighted_std * (n_eff ** (-1.0 / 5.0))
        kernel_values = np.exp(-0.5 * ((point["a"] - samples) / bandwidth) ** 2) / (
            np.sqrt(2.0 * np.pi) * bandwidth
        )
        expected = np.sum(weights * kernel_values)

        actual = bn.posterior(
            query_node="a",
            evidence={},
            point=point,
            infer_dict=lw,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        # TODO: CLG query node case
        actual = bn.posterior(
            query_node="d",
            evidence={},
            point=point,
            # infer_dict=lw,
        )
        # np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_infer_gibbs(
        self, bn: SemiParametricBayesianNetwork, data: pd.DataFrame
    ) -> None:
        evidence = {"b": data.iloc[0]["b"]}
        n_samples = 250
        infer_dict = bn.infer(
            method="gibbs",
            evidence=evidence,
            n_samples=n_samples,
            n_chains=10,
            burn_in=20,
            seed=SEED,
        )

        assignments = infer_dict["parameters"]["assignments"]
        weights = infer_dict["parameters"]["weights"]
        acceptance_rate = infer_dict["parameters"]["acceptance_rate"]

        assert set(assignments.columns) == set(bn.nodes())
        assert len(assignments) == n_samples
        assert np.all(assignments["b"] == evidence["b"])

        # MCMC samples are equally weighted
        np.testing.assert_allclose(weights, 1.0 / n_samples, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12)

        # Only the CKDE nodes and their ancestors are sampled, the rest is integrated out
        assert set(acceptance_rate) == bn._sampled_nodes() - set(evidence)
        assert all(0.0 <= rate <= 1.0 for rate in acceptance_rate.values())

        # Nodes without CKDE children are always drawn exactly from their full conditional
        exact_rate = infer_dict["parameters"]["exact_rate"]
        assert set(exact_rate) == set(acceptance_rate)
        for node, rate in exact_rate.items():
            assert 0.0 <= rate <= acceptance_rate[node]
            if not any(bn.cpd(c).type() == pbn.CKDEType() for c in bn.children(node)):
                assert rate == 1.0

        # Non-evidence CLG nodes are Rao-Blackwellized: posterior mean per row, posterior std shared by all rows
        assert np.all(np.isfinite(assignments["d"].to_numpy(dtype=float)))
        assert infer_dict["parameters"]["posterior_std"]["d"] > 0

    def test_infer_gibbs_matches_LW(
        self, bn: SemiParametricBayesianNetwork, data: pd.DataFrame
    ) -> None:
        evidence = {"b": data.iloc[0]["b"]}
        lw = bn.infer(evidence=evidence, n_samples=2000, seed=SEED)
        gs = bn.infer(
            method="gibbs",
            evidence=evidence,
            n_samples=1000,
            n_chains=20,
            burn_in=30,
            seed=SEED,
        )

        lw_samples = lw["parameters"]["assignments"]["a"].to_numpy(dtype=float)
        lw_mean = np.sum(lw["parameters"]["weights"] * lw_samples)
        gs_mean = gs["parameters"]["assignments"]["a"].to_numpy(dtype=float).mean()

        np.testing.assert_allclose(gs_mean, lw_mean, atol=0.25 * np.std(lw_samples))

    def test_posterior_gibbs(
        self, bn: SemiParametricBayesianNetwork, data: pd.DataFrame
    ) -> None:
        # NOTE: KDE query node case
        point = pd.Series({"a": 1.5, "d": 2.0})
        samples = np.array([0.0, 1.0, 2.0, 4.0])
        weights = np.full(len(samples), 1.0 / len(samples))

        gs = {
            "evidence": {},
            "method": "gibbs",
            "parameters": {
                "assignments": pd.DataFrame({"a": samples, "d": samples}),
                "weights": weights,
                "posterior_std": {},
            },
        }

        bandwidth = 1.06 * np.std(samples) * (len(samples) ** (-1.0 / 5.0))
        kernel_values = np.exp(-0.5 * ((point["a"] - samples) / bandwidth) ** 2) / (
            np.sqrt(2.0 * np.pi) * bandwidth
        )
        expected = np.sum(weights * kernel_values)

        actual = bn.posterior(
            query_node="a",
            evidence={},
            point=point,
            infer_dict=gs,
            method="gibbs",
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

        # CLG query node case (smoke test)
        actual = bn.posterior(
            method="gibbs",
            query_node="d",
            evidence={},
            point=point,
            n_samples=200,
            n_chains=4,
            burn_in=30,
            seed=SEED,
        )
        assert np.isfinite(actual) and actual >= 0

    def test_conditional_mixture_matches_logl(
        self, bn: SemiParametricBayesianNetwork, data: pd.DataFrame
    ) -> None:
        points = data.iloc[:20].reset_index(drop=True)
        for node in bn.nodes():
            mixture_logl = bn._mixture_logl(bn._kernel_cache(node), node, points)
            np.testing.assert_allclose(
                mixture_logl, bn._node_logl(node, points), rtol=1e-8, atol=1e-8
            )

    def test_mixture_logl_extreme_points(self) -> None:
        # Far from the training data (|A|, |B| = 40) the mixture stays finite and still matches pybnesian
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()})
        rng = np.random.default_rng(SEED)
        points = pd.DataFrame(
            {node: rng.normal(0, 1, 40) for node in ["A", "B", "C", "D", "E", "F"]}
        )
        points.loc[20:, ["A", "B"]] = rng.choice([-1.0, 1.0], size=(20, 2)) * 40.0
        for node in ["B", "F"]:
            mixture_logl = bn._mixture_logl(bn._kernel_cache(node), node, points)
            pybnesian_logl = bn._node_logl(node, points)
            assert np.all(np.isfinite(mixture_logl))
            np.testing.assert_allclose(
                mixture_logl, pybnesian_logl, rtol=1e-8, atol=1e-8
            )

    @staticmethod
    def _two_node_bn(child_type: pbn.FactorType) -> SemiParametricBayesianNetwork:
        """x (bimodal CKDE) -> y (``child_type``)"""
        rng = np.random.default_rng(SEED)
        n = 300
        x = np.concatenate([rng.normal(-2, 0.5, n // 2), rng.normal(2, 0.7, n // 2)])
        y = 0.8 * x + np.sin(x) + 0.6 * rng.normal(size=n)
        bn = SemiParametricBayesianNetwork()
        bn.add_node("x")
        bn.add_node("y")
        bn.add_arc("x", "y")
        bn.set_node_type("x", pbn.CKDEType())
        bn.set_node_type("y", child_type)
        pbn.SemiparametricBN.fit(bn, pd.DataFrame({"x": x, "y": y}))
        bn.feature_names_in_ = ["x", "y"]
        return bn

    @staticmethod
    def _grid_posterior_moments(
        bn: SemiParametricBayesianNetwork, y_obs: float
    ) -> tuple[float, float]:
        """Brute-force mean and variance of p(x | y = y_obs) ∝ p(x) p(y_obs | x) on a grid"""
        grid = np.linspace(-6.0, 6.0, 4001)
        points = pd.DataFrame({"x": grid, "y": y_obs})
        log_p = bn._node_logl("x", points) + bn._node_logl("y", points)
        w = np.exp(log_p - logsumexp(log_p))
        mean = np.sum(w * grid)
        return mean, np.sum(w * (grid - mean) ** 2)

    @pytest.mark.parametrize(
        "child_type", [pbn.LinearGaussianCPDType(), pbn.CKDEType()], ids=["lg", "ckde"]
    )
    def test_infer_gibbs_matches_exact_posterior(
        self, child_type: pbn.FactorType
    ) -> None:
        bn = self._two_node_bn(child_type)
        for y_obs in [0.5, 3.0]:
            gs = bn.infer(
                method="gibbs",
                evidence={"y": y_obs},
                n_samples=4000,
                n_chains=200,
                burn_in=20,
                seed=SEED,
            )
            samples = gs["parameters"]["assignments"]["x"].to_numpy(dtype=float)
            exact_rate = gs["parameters"]["exact_rate"]["x"]
            if child_type == pbn.LinearGaussianCPDType():
                assert exact_rate == 1.0
            else:
                assert 0.0 < exact_rate <= 1.0

            expected_mean, expected_var = self._grid_posterior_moments(bn, y_obs)
            np.testing.assert_allclose(samples.mean(), expected_mean, atol=0.1)
            np.testing.assert_allclose(samples.var(), expected_var, rtol=0.15)

    # NOTE: Convergence tests of the Rao-Blackwellized inference on a small network with known structure.
    # Tolerances were calibrated to about 4-5 times the seed-to-seed standard deviation of the estimates.
    TOY_ARCS = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("A", "D"),
        ("C", "E"),
        ("D", "E"),
        ("A", "F"),
        ("B", "F"),
    ]

    @classmethod
    def _toy_bn(
        cls, node_types: dict[str, pbn.FactorType], with_label: bool = False
    ) -> SemiParametricBayesianNetwork:
        """A (LG) -> B -> C -> D -> E with A -> D and C -> E, and F a child of A and B.
        Nodes missing from ``node_types`` are LinearGaussian. With ``with_label``, a 0/1 label Y (child of C and E)
        is added as the network's ``true_label``, so it is a node but not one of the ``feature_names_in_``.
        """
        rng = np.random.default_rng(SEED)
        n = 500
        a = rng.normal(0, 1, n)
        b = np.tanh(a) + 0.3 * rng.standard_t(3, n)
        c = 1 + 2 * b + rng.normal(0, 0.5, n)
        d = -1 + 0.5 * c + 0.3 * a + rng.normal(0, 0.5, n)
        e = d - c + rng.normal(0, 0.5, n)
        f = np.sin(2 * b) + 0.2 * a + 0.3 * rng.normal(size=n)
        df = pd.DataFrame({"A": a, "B": b, "C": c, "D": d, "E": e, "F": f})
        features = list(df.columns)
        arcs = list(cls.TOY_ARCS)
        if with_label:
            df["Y"] = (c + e > 1.5).astype(float)
            arcs += [("C", "Y"), ("E", "Y")]
        bn = SemiParametricBayesianNetwork(true_label="Y")
        for node in df.columns:
            bn.add_node(node)
        for source, target in arcs:
            bn.add_arc(source, target)
        for node in df.columns:
            bn.set_node_type(node, node_types.get(node, pbn.LinearGaussianCPDType()))
        pbn.SemiparametricBN.fit(bn, df)
        bn.feature_names_in_ = features
        return bn

    @staticmethod
    def _reference_means(
        bn: SemiParametricBayesianNetwork,
        evidence: dict[str, float],
        n_samples: int = 200_000,
    ) -> dict[str, float]:
        """Brute-force posterior means with plain likelihood weighting (every node forward sampled, no Rao-Blackwellization)"""
        samples = pd.DataFrame(index=np.arange(n_samples))
        log_weights = np.zeros(n_samples)
        for node_index, node in enumerate(bn.graph().topological_sort()):
            if node in evidence:
                samples[node] = float(evidence[node])
                log_weights += bn._node_logl(node, samples)
            else:
                samples[node] = (
                    bn.cpd(node)
                    .sample(
                        n_samples,
                        bn._parent_values(node, samples),
                        seed=SEED + node_index,
                    )
                    .to_pandas()
                    .to_numpy(dtype=float)
                )
        weights = np.exp(log_weights - logsumexp(log_weights))
        return {
            node: float(np.sum(weights * samples[node]))
            for node in samples.columns
            if node not in evidence
        }

    def test_infer_all_lg_matches_exact(self) -> None:
        # Without CKDE nodes nothing is sampled, so the Rao-Blackwellized posterior is exact
        bn = self._toy_bn({})
        nodes = list(bn.graph().topological_sort())
        index = {node: i for i, node in enumerate(nodes)}
        intercepts = np.zeros(len(nodes))
        variances = np.zeros(len(nodes))
        coefficients = np.zeros((len(nodes), len(nodes)))
        for node in nodes:
            cpd = bn.cpd(node)
            intercepts[index[node]] = cpd.beta[0]
            variances[index[node]] = cpd.variance
            for parent, coef in zip(cpd.evidence(), cpd.beta[1:]):
                coefficients[index[node], index[parent]] = coef
        A = np.linalg.inv(np.eye(len(nodes)) - coefficients)
        joint_mean = pd.Series(A @ intercepts, index=nodes)
        joint_cov = pd.DataFrame((A * variances) @ A.T, index=nodes, columns=nodes)

        evidence = {"B": 0.5, "E": 0.5}
        infer_dict = bn.infer(evidence=evidence, n_samples=10, seed=SEED)
        for query_node in ["A", "C", "D", "F"]:
            cond_mean, cond_cov = _conditional_gaussian(
                joint_mean, joint_cov, [query_node], evidence
            )
            point = pd.Series({query_node: cond_mean[0] + 0.3})
            expected = norm.pdf(
                point[query_node], loc=cond_mean[0], scale=np.sqrt(cond_cov[0, 0])
            )
            actual = bn.posterior(query_node, evidence, point, infer_dict=infer_dict)
            np.testing.assert_allclose(actual, expected, rtol=1e-8)

    def test_rao_blackwellize_clg_evidence_logl(self) -> None:
        # Several Rao-Blackwellized evidence nodes on rows with different sampled values (regression: NaN log-likelihoods)
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()})
        sampled = bn._sampled_nodes()
        assert sampled == {"A", "B", "F"}

        evidence = {"C": 3.0, "D": 2.0}
        assignments = pd.DataFrame(
            {"A": [0.1, 0.5, -1.0], "B": [0.0, 0.7, -1.2], "F": [0.2, 0.9, -0.4]}
        )
        log_lik, _, posterior_std = bn._rao_blackwellize_clg(
            assignments.copy(), evidence, sampled
        )

        # E is not observed, so log p(C, D | A, B) = log p(C | B) + log p(D | C, A)
        cpd_c, cpd_d = bn.cpd("C"), bn.cpd("D")
        beta_d = dict(zip(cpd_d.evidence(), cpd_d.beta[1:]))
        expected = norm.logpdf(
            evidence["C"],
            loc=cpd_c.beta[0] + cpd_c.beta[1] * assignments["B"],
            scale=np.sqrt(cpd_c.variance),
        ) + norm.logpdf(
            evidence["D"],
            loc=cpd_d.beta[0]
            + beta_d["C"] * evidence["C"]
            + beta_d["A"] * assignments["A"],
            scale=np.sqrt(cpd_d.variance),
        )
        assert np.all(np.isfinite(log_lik))
        np.testing.assert_allclose(log_lik, expected, rtol=1e-10)
        assert set(posterior_std) == {"E"}

    @pytest.mark.parametrize(
        "evidence",
        [{"D": 2.0, "E": 0.5}, {"F": 0.9, "D": 1.0}],
        ids=["lg_evidence", "ckde_and_lg_evidence"],
    )
    def test_infer_converges_to_reference(self, evidence: dict[str, float]) -> None:
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()})
        expected = self._reference_means(bn, evidence)
        estimates = []
        for run in range(4):
            infer_dict = bn.infer(
                evidence=evidence, n_samples=20000, seed=SEED + 100 * run
            )
            weights = infer_dict["parameters"]["weights"]
            assignments = infer_dict["parameters"]["assignments"]
            estimates.append([np.sum(weights * assignments[node]) for node in "ABC"])
        np.testing.assert_allclose(
            np.mean(estimates, axis=0),
            [expected[node] for node in "ABC"],
            atol=0.05,
        )

    def test_infer_gibbs_converges_to_reference(self) -> None:
        # Evidence on the CKDE node F exercises the rejection sampling and Metropolis-Hastings fallback of A and B
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()})
        evidence = {"F": 0.9, "D": 1.0}
        expected = self._reference_means(bn, evidence)
        estimates = []
        for run in range(2):
            gs = bn.infer(
                method="gibbs",
                evidence=evidence,
                n_samples=1000,
                n_chains=100,
                burn_in=30,
                seed=SEED + run,
            )
            assert gs["parameters"]["exact_rate"]["A"] < 1.0
            assignments = gs["parameters"]["assignments"]
            estimates.append([assignments[node].mean() for node in "ABC"])
        np.testing.assert_allclose(
            np.mean(estimates, axis=0),
            [expected[node] for node in "ABC"],
            atol=0.05,
        )

    def test_infer_gibbs_uses_ckde_evidence_on_lg_parents(self) -> None:
        # Regression: the Rao-Blackwellized LG parent A of the observed CKDE node B must not stay at its prior
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()})
        evidence = {"B": 0.9}
        lw = bn.infer(evidence=evidence, n_samples=20000, seed=SEED)
        lw_mean = np.sum(
            lw["parameters"]["weights"] * lw["parameters"]["assignments"]["A"]
        )
        gs = bn.infer(
            method="gibbs",
            evidence=evidence,
            n_samples=1000,
            n_chains=20,
            burn_in=30,
            seed=SEED,
        )
        gs_mean = gs["parameters"]["assignments"]["A"].mean()

        prior_mean = bn.cpd("A").beta[0]
        assert abs(lw_mean - prior_mean) > 0.5
        np.testing.assert_allclose(gs_mean, lw_mean, atol=0.1)

    def test_infer_effective_sample_size(self, tmp_path: Path) -> None:
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()})
        n_samples = 500

        infer_dict = bn.infer(evidence={}, n_samples=n_samples, seed=SEED)
        np.testing.assert_allclose(
            infer_dict["parameters"]["effective_sample_size"], n_samples
        )

        json_file_path = tmp_path / "infer_result.json"
        with pytest.warns(RuntimeWarning, match="Effective sample size"):
            infer_dict = bn.infer(
                evidence={"D": 2.0},
                n_samples=n_samples,
                seed=SEED,
                json_file_path=json_file_path,
            )
        weights = infer_dict["parameters"]["weights"]
        ess = infer_dict["parameters"]["effective_sample_size"]
        np.testing.assert_allclose(ess, 1.0 / np.sum(weights**2))
        assert 1.0 <= ess < n_samples

        # The export must not modify the returned dictionary
        assert isinstance(weights, np.ndarray)
        assert isinstance(infer_dict["parameters"]["assignments"], pd.DataFrame)
        with open(json_file_path) as f:
            exported = json.load(f)
        assert exported["parameters"]["effective_sample_size"] == ess

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_posterior_validation(self) -> None:
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()})
        point = pd.Series({"A": 0.0, "D": 0.0})
        infer_dict = bn.infer(evidence={"D": 2.0}, n_samples=50, seed=SEED)

        with pytest.raises(ValueError, match="disjoint"):
            bn.posterior("D", {"D": 2.0}, point, infer_dict=infer_dict)
        with pytest.raises(ValueError, match="different evidence"):
            bn.posterior("A", {"D": 1.0}, point, infer_dict=infer_dict)

    def test_mcmc_diagnostics(self) -> None:
        rng = np.random.default_rng(SEED)
        n_draws, n_chains = 2000, 8

        # Independent draws: R-hat ~ 1 and ESS ~ number of draws
        iid = rng.normal(size=(n_draws, n_chains))
        np.testing.assert_allclose(
            SemiParametricBayesianNetwork._split_rhat(iid), 1.0, atol=0.01
        )
        np.testing.assert_allclose(
            SemiParametricBayesianNetwork._mcmc_effective_sample_size(iid),
            n_draws * n_chains,
            rtol=0.15,
        )

        # AR(1) chains x_t = phi x_{t-1} + eps_t: ESS ~ N (1 - phi) / (1 + phi)
        phi = 0.8
        ar = np.zeros((n_draws, n_chains))
        ar[0] = rng.normal(size=n_chains) / np.sqrt(1 - phi**2)
        for t in range(1, n_draws):
            ar[t] = phi * ar[t - 1] + rng.normal(size=n_chains)
        np.testing.assert_allclose(
            SemiParametricBayesianNetwork._mcmc_effective_sample_size(ar),
            n_draws * n_chains * (1 - phi) / (1 + phi),
            rtol=0.25,
        )

        # Chains stuck around different values have R-hat >> 1
        stuck = iid + np.arange(n_chains)[None, :]
        assert SemiParametricBayesianNetwork._split_rhat(stuck) > 1.5

        # Tail ESS of independent draws ~ number of draws
        np.testing.assert_allclose(
            SemiParametricBayesianNetwork._tail_effective_sample_size(iid),
            n_draws * n_chains,
            rtol=0.15,
        )

        # Too few draws per chain, or a single chain for R-hat
        assert np.isnan(SemiParametricBayesianNetwork._split_rhat(iid[:3]))
        assert np.isnan(SemiParametricBayesianNetwork._split_rhat(iid[:, :1]))
        assert np.isnan(
            SemiParametricBayesianNetwork._mcmc_effective_sample_size(iid[:3])
        )

    def test_mcmc_diagnostics_match_arviz(self) -> None:
        az = pytest.importorskip("arviz")
        rng = np.random.default_rng(SEED)
        ar = np.zeros((501, 4))
        ar[0] = rng.normal(size=4)
        for t in range(1, len(ar)):
            ar[t] = 0.9 * ar[t - 1] + rng.normal(size=4)
        draws_cases = {
            "iid_odd_draws": rng.normal(size=(1001, 6)),
            "autocorrelated": ar,
            "stuck_heavy_tails": rng.standard_t(2, size=(300, 5)) + np.arange(5),
            "ties": np.round(rng.normal(size=(200, 4)), 1),
        }
        for name, draws in draws_cases.items():
            # arviz expects (n_chains, n_draws)
            np.testing.assert_allclose(
                SemiParametricBayesianNetwork._split_rhat(draws),
                float(az.rhat(draws.T)),
                rtol=1e-10,
                err_msg=name,
            )
            np.testing.assert_allclose(
                SemiParametricBayesianNetwork._mcmc_effective_sample_size(draws),
                float(az.ess(draws.T, method="bulk")),
                rtol=1e-10,
                err_msg=name,
            )
            np.testing.assert_allclose(
                SemiParametricBayesianNetwork._tail_effective_sample_size(draws),
                float(az.ess(draws.T, method="tail")),
                rtol=1e-10,
                err_msg=name,
            )

    def test_infer_gibbs_diagnostics(self) -> None:
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()})
        evidence = {"F": 0.9, "D": 1.0}
        n_samples, n_chains = 1000, 50
        gs = bn.infer(
            method="gibbs",
            evidence=evidence,
            n_samples=n_samples,
            n_chains=n_chains,
            burn_in=30,
            seed=SEED,
        )
        assert gs["evidence"] == evidence
        rhat = gs["parameters"]["rhat"]
        ess = gs["parameters"]["effective_sample_size"]
        tail_ess = gs["parameters"]["tail_effective_sample_size"]
        free_nodes = set(bn.nodes()) - set(evidence)
        assert set(rhat) == free_nodes and set(ess) == free_nodes
        assert set(tail_ess) == free_nodes

        # n_samples is a multiple of n_chains, so the diagnostics are those of the returned assignments (posterior
        # means for the Rao-Blackwellized nodes C and E), whose rows are ordered sweep by sweep
        assignments = gs["parameters"]["assignments"]
        assert set(gs["parameters"]["posterior_std"]) == {"C", "E"}
        for node in free_nodes:
            draws = assignments[node].to_numpy(dtype=float).reshape(-1, n_chains)
            assert rhat[node] == bn._split_rhat(draws)
            assert ess[node] == bn._mcmc_effective_sample_size(draws)
            assert tail_ess[node] == bn._tail_effective_sample_size(draws)
            # R-hat is close to 1 (the exact value depends on the chain length relative to the autocorrelation)
            assert 0.95 < rhat[node] < 1.5
            # The ESS can exceed the number of draws (antithetic chains), up to n log10(n)
            assert 0.0 < ess[node] <= n_samples * np.log10(n_samples)

        # B and C are strongly dependent (C = 1 + 2 B + small noise), which made Gibbs sampling mix slowly before C was
        # integrated out of the chains (R-hat ~1.1-1.2 with 20 draws per chain)
        assert rhat["B"] < 1.05 and rhat["C"] < 1.05

        # posterior with method="gibbs" uses the autocorrelation-aware ESS of the query node for the KDE bandwidth
        point = pd.Series({"A": 1.0})
        expected = bn._posterior_density(
            "A",
            point,
            gs["parameters"]["assignments"],
            gs["parameters"]["weights"],
            gs["parameters"]["posterior_std"],
            effective_sample_size=ess["A"],
        )
        actual = bn.posterior("A", evidence, point, infer_dict=gs, method="gibbs")
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_posterior_gibbs_validation(self) -> None:
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()})
        point = pd.Series({"A": 0.0, "D": 0.0})
        gs = bn.infer(
            method="gibbs",
            evidence={"D": 2.0},
            n_samples=40,
            n_chains=10,
            burn_in=5,
            seed=SEED,
        )

        with pytest.raises(ValueError, match="disjoint"):
            bn.posterior("D", {"D": 2.0}, point, infer_dict=gs, method="gibbs")
        with pytest.raises(ValueError, match="different evidence"):
            bn.posterior("A", {"D": 1.0}, point, infer_dict=gs, method="gibbs")
        with pytest.raises(ValueError, match="different method"):
            bn.posterior("A", {"D": 2.0}, point, infer_dict=gs, method="lw")
        with pytest.raises(ValueError, match="Unknown inference method"):
            bn.infer(evidence={"D": 2.0}, method="mcmc")

    def test_infer_discrete_parent(self) -> None:
        # Discrete C -> X <- K (CKDE): X is a CLinearGaussianCPD, Rao-Blackwellized with the parameters of the sampled class
        rng = np.random.default_rng(SEED)
        n = 4000
        c = rng.choice(["u", "v"], size=n, p=[0.3, 0.7])
        k = rng.normal(0, 1, n)
        x = np.where(c == "u", 1.0, -2.0) + 0.5 * k + rng.normal(0, 1, n)
        df = pd.DataFrame({"C": pd.Categorical(c), "K": k, "X": x})
        bn = SemiParametricBayesianNetwork(true_label="C")
        for node in df.columns:
            bn.add_node(node)
        bn.add_arc("C", "X")
        bn.add_arc("K", "X")
        bn.set_node_type("C", pbn.DiscreteFactorType())
        bn.set_node_type("K", pbn.CKDEType())
        bn.set_node_type("X", pbn.LinearGaussianCPDType())
        pbn.SemiparametricBN.fit(bn, df)

        # Exact p(x | k, C) and the class prior
        k_value, x_value = 0.4, -0.5
        prior = {value: np.mean(c == value) for value in "uv"}
        density = {}
        for value in "uv":
            cpd = bn.cpd("X").conditional_factor(pbn.Assignment({"C": value}))
            density[value] = norm.pdf(
                x_value,
                loc=cpd.beta[0] + cpd.beta[1] * k_value,
                scale=np.sqrt(cpd.variance),
            )
        marginal = sum(prior[value] * density[value] for value in "uv")
        point = pd.Series({"X": x_value})

        # p(x | k) is the mixture over the sampled classes, and p(x | k, C) is exact
        actual = bn.posterior("X", {"K": k_value}, point, n_samples=20000, seed=SEED)
        np.testing.assert_allclose(actual, marginal, rtol=0.02)
        actual = bn.posterior("X", {"K": k_value, "C": "u"}, point, n_samples=50)
        np.testing.assert_allclose(actual, density["u"], rtol=1e-10)

        # With evidence on X, the weights give P(C | x, k)
        infer_dict = bn.infer(
            evidence={"K": k_value, "X": x_value}, n_samples=20000, seed=SEED
        )
        weights = infer_dict["parameters"]["weights"]
        is_u = (infer_dict["parameters"]["assignments"]["C"] == "u").to_numpy()
        np.testing.assert_allclose(
            weights[is_u].sum(), prior["u"] * density["u"] / marginal, rtol=0.05
        )

        with pytest.raises(ValueError, match="not a category"):
            bn.infer(evidence={"C": "w"})
        with pytest.raises(NotImplementedError, match="discrete nodes"):
            bn.infer(evidence={"K": k_value}, method="gibbs")

    def test_infer_label_node(self) -> None:
        # The label Y is a node of the network but not one of the feature_names_in_
        bn = self._toy_bn({}, with_label=True)
        assert "Y" in bn.nodes() and "Y" not in bn.feature_names_in_
        evidence = {"B": 0.5, "D": 2.0}

        # Without CKDE nodes both methods are exact (nothing is sampled)
        nodes = list(bn.graph().topological_sort())
        index = {node: i for i, node in enumerate(nodes)}
        intercepts = np.zeros(len(nodes))
        variances = np.zeros(len(nodes))
        coefficients = np.zeros((len(nodes), len(nodes)))
        for node in nodes:
            cpd = bn.cpd(node)
            intercepts[index[node]] = cpd.beta[0]
            variances[index[node]] = cpd.variance
            for parent, coef in zip(cpd.evidence(), cpd.beta[1:]):
                coefficients[index[node], index[parent]] = coef
        A = np.linalg.inv(np.eye(len(nodes)) - coefficients)
        joint_mean = pd.Series(A @ intercepts, index=nodes)
        joint_cov = pd.DataFrame((A * variances) @ A.T, index=nodes, columns=nodes)
        cond_mean, cond_cov = _conditional_gaussian(
            joint_mean, joint_cov, ["Y"], evidence
        )
        point = pd.Series({"Y": cond_mean[0] + 0.1})
        expected = norm.pdf(point["Y"], loc=cond_mean[0], scale=np.sqrt(cond_cov[0, 0]))
        np.testing.assert_allclose(
            bn.posterior("Y", evidence, point, n_samples=10), expected, rtol=1e-8
        )
        np.testing.assert_allclose(
            bn.posterior(
                "Y", evidence, point, n_samples=40, n_chains=10, method="gibbs"
            ),
            expected,
            rtol=1e-8,
        )

        # With CKDE nodes, the label is Rao-Blackwellized by both methods
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()}, with_label=True)
        evidence = {"F": 0.9, "D": 1.0}
        lw = bn.infer(evidence=evidence, n_samples=5000, seed=SEED)
        gs = bn.infer(
            method="gibbs",
            evidence=evidence,
            n_samples=500,
            n_chains=50,
            burn_in=30,
            seed=SEED,
        )
        for result in (lw, gs):
            assert result["parameters"]["posterior_std"]["Y"] > 0
            assert np.all(
                np.isfinite(result["parameters"]["assignments"]["Y"].to_numpy(float))
            )
        lw_mean = np.sum(
            lw["parameters"]["weights"] * lw["parameters"]["assignments"]["Y"]
        )
        gs_mean = gs["parameters"]["assignments"]["Y"].mean()
        np.testing.assert_allclose(gs_mean, lw_mean, atol=0.05)

    def test_inference_warnings_and_json(self, tmp_path: Path) -> None:
        bn = self._toy_bn({"B": pbn.CKDEType(), "F": pbn.CKDEType()})

        # Degenerate likelihood weights
        with pytest.warns(RuntimeWarning, match="Effective sample size"):
            bn.infer(evidence={"F": 0.9}, n_samples=50, seed=SEED)

        # Too few Gibbs samples (such short chains also have a high R-hat)
        with (
            pytest.warns(RuntimeWarning, match="Effective sample size below"),
            pytest.warns(RuntimeWarning, match="R-hat above"),
        ):
            bn.infer(
                method="gibbs",
                evidence={"F": 0.9, "D": 1.0},
                n_samples=40,
                n_chains=5,
                burn_in=5,
                seed=SEED,
            )
        # Unconverged chains (the collapsed sampler converges too fast to trigger it reliably end to end)
        with pytest.warns(RuntimeWarning, match="R-hat above"):
            bn._warn_diagnostics({"A": 1.2, "B": 1.0}, {"A": 500.0, "B": 500.0})

        # Fewer than 4 draws per chain: NaN diagnostics, exported as JSON null
        json_file_path = tmp_path / "gs_result.json"
        gs = bn.infer(
            method="gibbs",
            evidence={"D": 1.0},
            n_samples=20,
            n_chains=10,
            burn_in=5,
            seed=SEED,
            json_file_path=json_file_path,
        )
        assert np.isnan(gs["parameters"]["rhat"]["A"])

        def _reject_constant(constant: str):
            raise ValueError(f"Non-standard JSON constant {constant}")

        with open(json_file_path) as f:
            exported = json.loads(f.read(), parse_constant=_reject_constant)
        assert exported["parameters"]["rhat"]["A"] is None
