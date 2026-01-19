from typing import Sequence

import pandas as pd
import pybnesian as pbn
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

from .constants import N_NEIGHBORS


class Graph:
    """
    A class to represent an undirected, weighted graph and provide utility methods for graph operations,
    including adding edges and computing the Minimum Spanning Tree (MST) using Kruskal's algorithm.
        graph (dict): A dictionary representing the adjacency list of the graph, where each key is a node and the value is a dictionary of neighboring nodes and edge weights.
        result (list): A list to store the edges included in the result of the last MST computation.
    Methods:
        __init__():
            Initializes an empty graph and result list.
        add_edge(u: str, v: str, w: float) -> None:
            Adds an undirected edge between nodes `u` and `v` with weight `w`.
        add_edges(edges: Sequence[tuple[str, str, float]]) -> None:
            Adds multiple edges to the graph from a sequence of (u, v, w) tuples.
        get_edges() -> list[tuple[str, str, float]]:
            Retrieves all edges in the graph as a list of (source_node, target_node, weight) tuples.
        kruskal_mst() -> float:
            Computes the Minimum Spanning Tree (MST) of the graph using Kruskal's algorithm.
            Returns the total weight of the MST and logs the edges included.
        _find(parent: dict[str, str], i: str) -> str:
            Finds the root of the set containing node `i` with path compression.
        _union(parent: dict[str, str], rank: dict[str, int], x: str, y: str) -> None:
            Unites the sets containing nodes `x` and `y` using union by rank.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the class.

        Attributes:
            graph (dict): A dictionary to store the graph structure.
            result (list): A list to store the results.
        """
        self.graph = {}
        self.result = []

    def add_edge(self, u: str, v: str, w: float) -> None:
        """
        Adds an edge between two nodes in the graph with a given weight.

        Parameters:
        u (str): The first node.
        v (str): The second node.
        w (float): The weight of the edge.

        Returns:
        None
        """
        if u not in self.graph:
            self.graph[u] = {}
        if v not in self.graph:
            self.graph[v] = {}
        self.graph[u][v] = w
        self.graph[v][u] = w  # For undirected graph

    def add_edges(self, edges: Sequence[tuple[str, str, float]]) -> None:
        """
        Add multiple edges to the graph.

        Parameters:
        edges (Sequence[tuple[str, str, float]]): A sequence of tuples where each tuple represents an edge.
            Each tuple contains three elements:
            - u (str): The starting node of the edge.
            - v (str): The ending node of the edge.
            - w (float): The weight of the edge.

        Returns:
        None
        """
        for u, v, w in edges:
            self.add_edge(u, v, w)

    def get_edges(self) -> list[tuple[str, str, float]]:
        """
        Retrieve all edges from the graph.

        Returns:
            list[tuple[str, str, float]]: A list of tuples where each tuple represents an edge in the format (source_node, target_node, weight).
        """
        edges = []
        for u in self.graph:
            for v in self.graph[u]:
                edges.append((u, v, self.graph[u][v]))
        return edges

    def kruskal_mst(self) -> float:
        """
        Main function to perform Kruskal's algorithm to find the Minimum Spanning Tree (MST) of a graph.
        # SEE: https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
        Returns:
            float: The total weight of the MST.

        The function follows these steps:
        1. Sort all the edges in non-decreasing order of their weight.
        2. Create subsets for each vertex.
        3. Iterate through the sorted edges and pick the smallest edge. If it does not form a cycle, include it in the result.
        4. Repeat until there are V-1 edges in the result (where V is the number of vertices).

        The function also prints the edges included in the MST and their weights, as well as the total weight of the MST.
        """

        # This will store the resultant MST
        result = []

        # An index variable, used for sorted edges
        i = 0

        # An index variable, used for result[]
        e = 0

        # Sort all the edges in
        # non-decreasing order of their
        # weight
        edges = self.get_edges()
        edges = sorted(edges, key=lambda item: item[2])

        parent = {}
        rank = {}

        # Create V subsets with single elements
        nodes = set(u for u, v, w in edges).union(v for u, v, w in edges)
        for node in nodes:
            parent[node] = node
            rank[node] = 0

        # Number of edges to be taken is less than to V-1
        while e < len(nodes) - 1:

            # Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = edges[i]
            i = i + 1
            x = self._find(parent, u)
            y = self._find(parent, v)

            # If including this edge doesn't
            # cause cycle, then include it in result
            # and increment the index of result
            # for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self._union(parent, rank, x, y)
            # Else discard the edge

        minimum_cost = 0
        # Module.log_note(
        #     3,
        #     "Edges in the constructed MST",
        #     category="DEBUG",
        # )
        for u, v, weight in result:
            minimum_cost += weight
            # Module.log_note(
            #     3,
            #     f"{u} -- {v} == {weight}",
            #     category="DEBUG",
            # )
            # Module.log_note(
            #     3,
            #     f"Minimum Spanning Tree: {minimum_cost}",
            #     category="DEBUG",
            # )

        self.result = result
        return minimum_cost

    def _find(self, parent: dict[str, str], i: str) -> str:
        """
        A utility function to find the root of the set in which element i is present.
        Implements path compression to flatten the structure of the tree, making future queries faster.

        Args:
            parent (dict[str, str]): A dictionary where each key is a node and the value is the parent of that node.
            i (str): The node for which the root is to be found.

        Returns:
            str: The root of the set in which element i is present.
        """
        if parent[i] != i:

            # Reassignment of node's parent
            # to root node as
            # path compression requires
            parent[i] = self._find(parent, parent[i])
        return parent[i]

    def _union(
        self, parent: dict[str, str], rank: dict[str, int], x: str, y: str
    ) -> None:
        """
        Perform the union of two sets x and y using the union by rank heuristic.

        Args:
            parent (dict[str, str]): A dictionary representing the parent pointers of the disjoint-set forest.
            rank (dict[str, int]): A dictionary representing the rank (approximate tree height) of each element.
            x (str): The representative (root) of the first set.
            y (str): The representative (root) of the second set.

        Returns:
            None
        """

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[y] = x
            rank[x] += 1


class DirectedTree(Graph):
    """
    DirectedTree is a subclass of Graph that represents a directed tree structure.
    This class allows for the creation and manipulation of a directed tree, where each edge has a direction and an associated weight.
    The tree is constructed from a specified root node and a list of weighted edges. The class provides methods to retrieve all edges
    in the tree and to insert undirected edges using a breadth-first search (BFS) traversal starting from the root.
    Attributes:
        graph (dict): A dictionary representing the adjacency list of the tree, where keys are node identifiers and values are lists
            of tuples (child_node, weight).
    Methods:
        __init__(root: str, edges: list[tuple[str, str, float]]) -> None:
            Initializes the DirectedTree with a root node and a list of weighted edges.
        get_edges() -> list[tuple[str, str, float]]:
            Returns a list of all edges in the tree as tuples (parent_node, child_node, weight).
        _insert_undirected_edges(root: str, edges: list[tuple[str, str, float]]) -> None:
            Private method to insert undirected edges into the tree using BFS traversal from the root node.
    """

    def __init__(self, root: str, edges: list[tuple[str, str, float]]) -> None:
        """
        Initializes the instance of the class.

        This constructor calls the parent class's initializer and sets up an empty dictionary `tree`
        to store the tree structure.

        Returns:
            None
        """
        super().__init__()
        self._insert_undirected_edges(root, edges)

    def get_edges(self) -> list[tuple[str, str, float]]:
        """
        Retrieve all edges from the tree.

        This method iterates through the tree structure and collects all edges
        in the form of tuples containing the parent node, child node, and the
        weight of the edge.

        Returns:
            list[tuple[str, str, float]]: A list of tuples where each tuple
            represents an edge in the format (parent_node, child_node, weight).
        """
        edges = []
        for node in self.graph:
            for child, weight in self.graph[node]:
                edges.append((node, child, weight))
        return edges

    def _insert_undirected_edges(
        self, root: str, edges: list[tuple[str, str, float]]
    ) -> None:
        """
        Insert undirected edges into the tree starting from the root node.

        This method takes a list of edges and a root node, and constructs an undirected
        graph by adding edges to the tree. It uses a breadth-first search (BFS) approach
        to traverse the graph and insert edges.

        Args:
            edges (list[tuple[str, str, float]]): A list of edges where each edge is represented
                as a tuple (u, v, w) with u and v being the nodes and w being the weight of the edge.
            root (str): The root node from which to start the BFS traversal.

        Returns:
            None
        """
        self.graph[root] = []
        visited = set([root])
        queue = [root]

        while queue:
            node = queue.pop(0)
            for edge in edges:
                u, v, w = edge
                if u == node and v not in visited:
                    self.graph.setdefault(node, []).append((v, w))
                    self.graph.setdefault(v, [])
                    visited.add(v)
                    queue.append(v)
                elif v == node and u not in visited:
                    self.graph.setdefault(node, []).append((u, w))
                    self.graph.setdefault(u, [])
                    visited.add(u)
                    queue.append(u)


class ConditionalMutualInformationMatrix:
    """
    A class for computing mutual information matrices between variables in a dataset,
    optionally conditioned on another variable.
    This class provides methods to calculate pairwise mutual information between variables
    in a pandas DataFrame, as well as conditional mutual information weighted by the
    probability of a specified conditioning variable's values. It supports both unconditional
    and conditional mutual information calculations using k-nearest neighbors estimation.
    Attributes:
        n_neighbors (int): The number of neighbors to use for k-nearest neighbors mutual information estimation.
        k_mutual_info: An instance of the mutual information estimator used for calculations.
    Methods:
        __init__(n_neighbors: int) -> None:
            Initializes the ConditionalMutualInformationMatrix with the specified number of neighbors.
        calculate_continuous(X: pd.DataFrame, y: pd.Series | None = None) -> list[tuple[str, str, float]]:
            Calculates the (conditional) mutual information between pairs of continuous nodes in the dataset.
        calculate_discrete(X: pd.DataFrame, y: pd.Series) -> list[tuple[str, str, float]]:
            Calculates the conditional mutual information for a given discrete conditional variable.
            This function computes the mutual information between pairs of nodes in the dataset,
            conditioned on the values of a specified variable. The mutual information values are
            weighted by the probability of the conditional variable's values.
    """

    def __init__(self, n_neighbors: int) -> None:
        """
        Initializes the class with the specified number of neighbors.

        Args:
            n_neighbors (int): The number of neighbors to consider.
        """
        self.n_neighbors = n_neighbors
        self.k_mutual_info = None

    def calculate_continuous(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> list[tuple[str, str, float]]:
        """Calculate the (conditional) mutual information between pairs of CONTINUOUS nodes in the dataset."""
        data = pd.concat([X, y], axis=1) if y is not None else X
        self.k_mutual_info = pbn.KMutualInformation(data, k=self.n_neighbors)
        edges = []
        for i, node in enumerate(X.columns):
            for j in range(i + 1, len(X.columns)):
                node2 = X.columns[j]
                if y is not None:
                    mutual_info_value = self.k_mutual_info.mi(node, node2, y.name)
                else:
                    mutual_info_value = self.k_mutual_info.mi(node, node2)
                edge = (node, node2, mutual_info_value)
                edges.append(edge)
        return edges

    def calculate_discrete(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list[tuple[str, str, float]]:
        """
        Calculate the conditional mutual information between CONTINUOUS nodes given a DISCRETE conditional variable.

        This function computes the mutual information between pairs of nodes in the dataset,
        conditioned on the values of a specified variable. The mutual information values are
        weighted by the probability of the conditional variable's values.

        Args:
            X (pd.DataFrame): The input data as a pandas DataFrame.
            y (pd.Series): The target variable as a pandas Series, which is used to condition
            on the mutual information calculations.
            The values in `y` are used to compute the conditional probability distribution.

        Returns:
            list[tuple[str, str, float]]: A list of tuples where each tuple contains two nodes
            and their corresponding weighted mutual information value.
        """

        conditional_probability = y.value_counts(normalize=True)
        edges_df_list = []
        for value, prob in conditional_probability.items():
            data_subset = X[y == value]
            # RFE: Do this for categorical variables only to fix DTAN
            # if all(data_subset.dtypes == "category"):
            #     self.k_mutual_info = pbn.ChiSquare(data_subset)
            # else:
            # RFE: Check if y can be set and used
            edges = self.calculate_continuous(data_subset, y=None)
            edges_df = pd.DataFrame(edges, columns=["node", "node2", "mutual_info"])
            edges_df.set_index(["node", "node2"], inplace=True)
            edges_df["mutual_info"] *= prob
            edges_df_list.append(edges_df)

        # Sum the mutual information values in each list
        edges_df = (
            pd.concat(edges_df_list).groupby(["node", "node2"]).sum().reset_index()
        )
        # Convert edges_df into list of tuples
        edges = list(edges_df.itertuples(index=False, name=None))
        return edges


# region Main class for Conditional Mutual Information Graph
# RFE: Implement (continuous) MutualInformationGraph
class ConditionalMutualInformationGraph(Graph):
    """
    A graph structure for computing maximum weighted spanning trees based on conditional mutual information.
    This class extends the base Graph class to support the calculation of maximum weighted spanning trees
    where edge weights are determined by the conditional mutual information between variables, given a specified
    conditional variable. It leverages a ConditionalMutualInformationMatrix to compute the conditional mutual information
    matrix and uses Kruskal's algorithm to construct the spanning tree.
        n_neighbors (int): Number of neighbors to consider when computing the mutual information matrix. Default is 3.
    Attributes:
        mutual_info_matrix (ConditionalMutualInformationMatrix): Instance used to compute conditional mutual information matrices.
    Methods:
        __init__(n_neighbors: int = N_NEIGHBORS) -> None:
            Initializes the ConditionalMutualInformationGraph with the specified number of neighbors.
        calculate_class_mutual_info(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
            Calculates mutual information between each feature in X and the target variable y.
        calculate_conditional_mutual_info(X: pd.DataFrame, y: pd.Series) -> list[tuple[str, str, float]]:
            Computes conditional mutual information between pairs of features in X, conditioned on y.
        calculate_maximum_weighted_spanning_tree(X: pd.DataFrame, y: pd.Series) -> float:
            Computes the maximum weighted spanning tree based on conditional mutual information.
            This method calculates the conditional mutual information matrix for the given data and conditional variable,
            constructs a maximum weighted spanning tree using Kruskal's algorithm, and returns the total weight
            of the spanning tree.
        This method is useful for identifying the most informative relationships between features in the context of a Bayesian
        network classifier, allowing for the construction of a directed tree that captures the dependencies between features.
    """

    def __init__(self, n_neighbors: int = N_NEIGHBORS) -> None:
        """
        Initializes the class with the specified number of neighbors for the mutual information matrix.

        Args:
            n_neighbors (int): The number of neighbors to consider for the mutual information matrix. Default is 3.
        """
        super().__init__()
        self.mutual_info_matrix = ConditionalMutualInformationMatrix(n_neighbors)

    def calculate_class_mutual_info(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, float]:
        if all(X.dtypes == "category"):  # discrete features
            mi_values = {col: float(mutual_info_score(X[col], y)) for col in X.columns}
        else:  # continuous features
            mi = mutual_info_classif(X, y, n_neighbors=N_NEIGHBORS)
            mi_values = {col: float(val) for col, val in zip(X.columns, mi)}

        return mi_values

    def calculate_conditional_mutual_info(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list[tuple[str, str, float]]:
        """
        Calculate the conditional mutual information for a given conditional variable.

        This function computes the mutual information between pairs of nodes in the dataset,
        conditioned on the values of a specified variable. The mutual information values are
        weighted by the probability of the conditional variable's values.

        Args:
            X (pd.DataFrame): The input data as a pandas DataFrame.
            y (pd.Series): The target variable as a pandas Series, which is used to condition
            on the mutual information calculations.
            The values in `y` are used to compute the conditional probability distribution.

        Returns:
            list[tuple[str, str, float]]: A list of tuples where each tuple contains two nodes
            and their corresponding weighted mutual information value.
        """
        edges = self.mutual_info_matrix.calculate_discrete(X, y)
        return edges

    # RFE: Implement using external https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.mst.maximum_spanning_tree.html
    def calculate_maximum_weighted_spanning_tree(
        self, X: pd.DataFrame, y: pd.Series
    ) -> float:
        """
        Calculate the maximum weighted spanning tree for a given dataset and conditional variable.
        This function computes the mutual information matrix for the given data and conditional variable,
        then constructs a maximum weighted spanning tree using Kruskal's algorithm.
        Args:
            X (pd.DataFrame): The input data for which the spanning tree is to be calculated.
            y (pd.Series): The conditional variable used to calculate the mutual information.
        Returns:
            float: The maximum cost of the weighted spanning tree.
        """
        edges = self.calculate_conditional_mutual_info(X, y)
        # We change the sign of the mutual information values to find the maximum weighted spanning tree
        edges = [(u, v, -w) for u, v, w in edges]
        # Compute the maximum spanning tree using Kruskal's algorithm
        self.add_edges(edges)
        maximum_cost = -self.kruskal_mst()
        return maximum_cost


# endregion
