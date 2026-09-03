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
from src.bn import (
    BayesianNetwork,
    DiscreteBayesianNetwork,
    GaussianBayesianNetwork,
    KDEBayesianNetwork,
    SemiParametricBayesianNetwork,
)


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

    def _conditional_gaussian(self, joint_mean, joint_cov, query_nodes, evidence):
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

    def test_infer(self, bn: GaussianBayesianNetwork, data: pd.DataFrame) -> None:
        query_nodes = bn.nodes()
        evidence = {"b": data.iloc[0]["b"]}
        infer_dict = self._infer_result(bn, evidence)

        # Compute expected conditional mean and covariance using the joint Gaussian parameters
        expected_mean, expected_cov = self._conditional_gaussian(
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
            "parameters": {
                "assignments": pd.DataFrame({"a": samples}),
                "weights": weights,
            }
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
            likelihood_weighting_dict=lw,
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
            "parameters": {
                "assignments": pd.DataFrame({"a": samples, "d": samples}),
                "weights": weights,
            }
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
            likelihood_weighting_dict=lw,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        # TODO: CLG query node case
        actual = bn.posterior(
            query_node="d",
            evidence={},
            point=point,
            # likelihood_weighting_dict=lw,
        )
        # np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
