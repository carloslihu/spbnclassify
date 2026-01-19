import networkx as nx
import pandas as pd
import pytest
from helpers.data import (
    DATA_SIZE,
    N_NEIGHBORS,
    SEED,
    TRUE_CLASS_LABEL,
    generate_normal_data,
    generate_normal_data_classification,
)
from src.spbnclassify.utils import (
    ConditionalMutualInformationGraph,
    ConditionalMutualInformationMatrix,
    Graph,
)


class TestConditionalMutualInformationMatrix:
    @pytest.fixture(scope="class")
    def continuous_data(
        self, data_size: int = DATA_SIZE, seed: int = SEED
    ) -> pd.DataFrame:
        """Generates a normal classification dataset.

        Args:
            data_size (pd.DataFrame): The normal dataset.
            seed (int, optional): The seed for random sampling. Defaults to 0.

        Returns:
            pd.DataFrame: The dataset.
        """
        return generate_normal_data(data_size, seed=seed)

    @pytest.fixture(scope="class")
    def classification_data(
        self, data_size: int = DATA_SIZE, seed: int = SEED
    ) -> pd.DataFrame:
        """Generates a normal classification dataset.

        Args:
            data_size (pd.DataFrame): The normal dataset.
            seed (int, optional): The seed for random sampling. Defaults to 0.

        Returns:
            pd.DataFrame: The dataset.
        """
        return generate_normal_data_classification(data_size, seed=seed)

    def test_calculate_continuous(self, continuous_data):
        mutual_info_matrix = ConditionalMutualInformationMatrix(n_neighbors=N_NEIGHBORS)
        # Test without conditional variable
        edges = mutual_info_matrix.calculate_continuous(continuous_data)
        assert len(edges) == 6
        # Test with conditional variable
        edges = mutual_info_matrix.calculate_continuous(
            continuous_data.drop(columns=["a"]),
            continuous_data["a"],
        )
        assert len(edges) == 3

    def test_calculate_discrete(self, classification_data):
        mutual_info_matrix = ConditionalMutualInformationMatrix(n_neighbors=N_NEIGHBORS)
        edges = mutual_info_matrix.calculate_discrete(
            classification_data.drop(columns=[TRUE_CLASS_LABEL]),
            classification_data[TRUE_CLASS_LABEL],
        )
        assert len(edges) == 3


class TestGraph:

    def test_kruskal_mst(self) -> None:
        """Test the Kruskal MST algorithm.

        Args:
            data (pd.DataFrame): The data.
        """
        g = Graph()
        edges = [
            ("a", "b", 10),
            ("a", "c", 6),
            ("a", "d", 5),
            ("b", "d", 15),
            ("c", "d", 4),
        ]
        g.add_edges(edges)

        # Function call
        mst = g.kruskal_mst()

        # Compare with networkx's minimum_spanning_tree
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        nx_mst_graph = nx.minimum_spanning_tree(G)
        nx_mst = sum(d["weight"] for u, v, d in nx_mst_graph.edges(data=True))
        assert nx_mst == mst == 19


# TODO: DirectedTree class tests
class TestConditionalMutualInformationGraph:
    @pytest.fixture(scope="class")
    def classification_data(
        self, data_size: int = DATA_SIZE, seed: int = SEED
    ) -> pd.DataFrame:
        """Generates a normal classification dataset.

        Args:
            data_size (pd.DataFrame): The normal dataset.
            seed (int, optional): The seed for random sampling. Defaults to 0.

        Returns:
            pd.DataFrame: The dataset.
        """
        return generate_normal_data_classification(data_size, seed=seed)

    # TODO:
    def test_calculate_maximum_weighted_spanning_tree(self, classification_data):
        mutual_info_graph = ConditionalMutualInformationGraph(n_neighbors=N_NEIGHBORS)
        # print(
        #     mutual_info_graph.calculate_maximum_weighted_spanning_tree(
        #         classification_data, conditional_variable=TRUE_CLASS_LABEL
        #     )
        # )
        # print(mutual_info_graph.result)
