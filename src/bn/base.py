import pickle
from abc import abstractmethod
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyagrum as gum
import pyagrum.lib.image as gumimage
import pyagrum.lib.notebook as gnb
import pybnesian as pbn
from scipy.optimize import differential_evolution
from src.utils.constants import TRUE_ANOMALY_LABEL
from src.utils.feature_selection import get_zero_variance_variables

from ..utils import NAN_LOGL_VALUE, NODE_TYPE_COLOR_MAP


class BayesianNetworkInterface:
    """Abstract Bayesian Network interface
    This class is a wrapper for the pybnesian and pyagrum library
    """

    bn_type = None

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network

        Returns:
            str: The string representation
        """
        return (
            f"Bayesian Network with {self.num_nodes()} nodes and {self.num_arcs()} arcs"
        )

    @abstractmethod
    def nodes(self) -> list[str]:
        """Returns the list of nodes in the Bayesian Network

        Returns:
            list[str]: list of node names
        """
        pass

    @abstractmethod
    def num_nodes(self) -> int:
        """Returns the number of nodes in the Bayesian Network

        Returns:
            int: Number of nodes
        """
        pass

    @abstractmethod
    def node_types(self) -> dict[str, pbn.FactorType]:
        """Returns the node types of the Bayesian Network

        Returns:
            dict[str, pbn.FactorType]: Dictionary of node types
        """
        pass

    @abstractmethod
    def arcs(self) -> list[tuple[str, str]]:
        """Returns the list of arcs in the Bayesian Network
        Returns:
            list[tuple[str, str]]: list of arcs
        """
        pass

    @abstractmethod
    def num_arcs(self) -> int:
        """Returns the number of arcs in the Bayesian Network

        Returns:
            int: Number of arcs
        """
        pass

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "BayesianNetworkInterface":
        pass

    @abstractmethod
    def sample(self, sample_size: int, seed: int | None = None) -> pd.DataFrame:
        """Samples the Bayesian Network

        Args:
            sample_size (int): The number of samples to generate
            seed (int | None, optional): The seed for the random number generator. Defaults to None.

        Returns:
            pd.DataFrame: The generated samples
        """
        pass

    @abstractmethod
    def save(self, model_file: Path) -> None:
        """Saves the model to a file

        Args:
            model_file (Path): The model file path
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_file: Path) -> "BayesianNetworkInterface":
        """Loads the model from a file

        Args:
            model_file (Path): File path

        Returns:
            BayesianNetwork: The model
        """
        pass


# NOTE: Multiple inheritance makes super() not working properly
class BayesianNetwork(pbn.BayesianNetwork, BayesianNetworkInterface):
    """Abstract Bayesian Network class
    This class is a wrapper for the pybnesian library
    """

    bn_type = pbn.BayesianNetworkType()
    search_operators = []

    def __init__(
        self,
        search_score: str = "validated-lik",
        arc_blacklist: list[tuple[str, str]] = [],
        arc_whitelist: list[tuple[str, str]] = [],
        type_blacklist: list[tuple[str, pbn.FactorType]] = [],
        type_whitelist: list[tuple[str, pbn.FactorType]] = [],
        callback: pbn.Callback = None,
        max_indegree: int = 0,
        max_iters: int = 2147483647,
        epsilon: int = 0,
        patience: int = 0,
        seed: int | None = None,
        num_folds: int = 5,
        test_holdout_ratio: float = 0.2,
        max_train_data_size: int = 0,
        verbose: bool = False,
        feature_names_in_: list[str] = [],
        n_features_in_: int = 0,
        true_label: str = TRUE_ANOMALY_LABEL,
        prediction_label: str = "binary_predicted_label",
    ) -> None:
        """Initializes the Bayesian Network with the nodes, arcs, node_types and the structure learning parameters

        Args:
            nodes (list[str], optional): list of nodes. Defaults to [].
            arcs (list[tuple[str, str]], optional): list of arcs. Defaults to [].
            node_types (list[tuple[str, pbn.FactorType]], optional): list of node types. Defaults to [].
            search_score (str, optional): Search score to be used for the structure learning. The possible scores ((validate_options.cpp)) are:
                - "bic" (Bayesian Information Criterion)
                - "bge" (Bayesian Gaussian equivalent)
                - "cv-lik" (Cross-Validated likelihood)
                - "holdout-lik" (Hold-out likelihood)
                - "validated-lik" (Validated likelihood with cross-validation). Defaults to "bic".
            arc_blacklist (list[tuple[str, str]], optional): Arc blacklist (forbidden arcs). Defaults to [].
            arc_whitelist (list[tuple[str, str]], optional): Arc whitelist (forced arcs). Defaults to [].
            type_blacklist (list[tuple[str, pbn.FactorType]], optional): Node type blacklist (forbidden node types). Defaults to [].
            type_whitelist (list[tuple[str, pbn.FactorType]], optional): Node type whitelist (forced node types). Defaults to [].
            callback (pbn.Callback, optional): Callback function to be called after each iteration. Defaults to None.
            max_indegree (int, optional): Maximum indegree allowed in the graph. Defaults to 0.
            max_iters (int, optional): Maximum number of search iterations. Defaults to 2147483647.
            epsilon (int, optional): Minimum delta score allowed for each operator. If the new operator is less than epsilon, the search process is stopped. Defaults to 0.
            patience (int, optional): The patience parameter (only used with pbn.ValidatedScore). Defaults to 0.
            seed (int | None, optional): Seed parameter of the score (if needed). Defaults to None.
            num_folds (int, optional): Number of folds for the CVLikelihood and ValidatedLikelihood scores. Defaults to 5.
            test_holdout_ratio (float, optional): Parameter for the HoldoutLikelihood and ValidatedLikelihood scores. Defaults to 0.2.
            max_train_data_size (int, optional): Maximum sample size to be used for the structure learning. Defaults to 0.
            verbose (bool, optional): If True the progress will be displayed, otherwise nothing will be displayed. Defaults to False.
            true_label (str, optional): The true label column name. Defaults to TRUE_ANOMALY_LABEL.
            prediction_label (str, optional): The predicted label column name. Defaults to "binary_predicted_label".
        """

        pbn.BayesianNetwork.__init__(self, type=self.bn_type, nodes=[])
        BayesianNetworkInterface.__init__(self)
        self.graphic = gum.BayesNet()

        self.search_score = search_score
        self.arc_blacklist = arc_blacklist
        self.arc_whitelist = arc_whitelist
        self.type_blacklist = type_blacklist
        self.type_whitelist = type_whitelist
        self.callback = callback
        self.max_indegree = max_indegree
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.patience = patience
        self.seed = seed
        self.num_folds = num_folds
        self.test_holdout_ratio = test_holdout_ratio
        self.max_train_data_size = max_train_data_size
        self.verbose = verbose
        self.include_cpd = True  # So that the factors are picklable
        # Predictor variables
        self.feature_names_in_ = feature_names_in_
        self.n_features_in_ = n_features_in_
        # Target variables
        self.true_label = true_label
        self.prediction_label = prediction_label

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network

        Returns:
            str: The string representation
        """
        return BayesianNetworkInterface.__str__(self)

    # NOTE: Add attributes where corresponding for __init__ pickling
    # "graphic",
    # "joint_gaussian_",
    # "max_logl_df",
    def __getstate__(self) -> dict:
        """Extends the pickle dump attributes

        Returns:
            dict: Pickle state
        """
        state = self.__dict__.copy()

        state.update(
            {
                "__nodes": self.nodes(),
                "__arcs": self.arcs(),
                # "__graph": self.graph(),
                # "__type": self.type(),
                # You can omit this line if type is homogeneous
                "__node_types": list(self.node_types().items()),
            }
        )
        # Makes factors picklable
        if self.include_cpd and self.fitted():
            factors = []
            for n in self.nodes():
                if self.cpd(n) is not None:
                    factors.append(self.cpd(n))
            state["__factors"] = factors
        return state

    def __setstate__(self, state: dict) -> None:
        """Extends the pickle load attributes

        Args:
            state (dict): Pickle state
        """
        # NOTE Call the parent constructor always in __setstate__ !
        init_params = {
            k: v for k, v in state.items() if k in self.__init__.__code__.co_varnames
        }
        self.__init__(**init_params)
        self._copy_bn_structure(state["__arcs"], state["__node_types"])
        if "__factors" in state:
            self.add_cpds(state["__factors"])
        self.__dict__.update(state)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pbn.BayesianNetwork:
        """Learns ONLY the structure of the Bayesian Network from the data."""
        # We undersample the data for the structure learning
        if self.max_train_data_size > 0 and len(X) > self.max_train_data_size:
            structure_X = X.sample(n=self.max_train_data_size, random_state=self.seed)
            if y is not None:
                structure_y = y.loc[structure_X.index]
            else:
                structure_y = None
        else:
            structure_X = X
            structure_y = y
        # Update feature names and counts
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        # We remove the variables with zero variance
        self._remove_zero_variance_nodes(structure_X)

        structure_X = structure_X[self.feature_names_in_]

        bn = self._fit_structure(structure_X, structure_y)
        bn = self._fit_node_types(bn, structure_X, structure_y)

        # Copies the structure to the current Bayesian Network
        if bn:
            arcs = bn.arcs()
            node_types = bn.node_types().items()
            # RFE: Adapt for when there are more discrete nodes apart from the target
            node_num_categories_dict = {}
            if y is not None:
                num_categories = len(y.cat.categories)
                if num_categories < 2:
                    num_categories = 2
                node_num_categories_dict[self.true_label] = num_categories
            self._copy_bn_structure(arcs, node_types, node_num_categories_dict)
        return self

    def logl(self, X: pd.DataFrame) -> np.ndarray:
        """Calculates the log-likelihood of the data

        Args:
            data (pd.DataFrame): The data to calculate the log-likelihood

        Returns:
            np.ndarray: The log-likelihood of the data
        """

        log_likelihood = np.zeros(X.shape[0])
        for node in self.nodes():
            cpd = self.cpd(node)
            if cpd:
                # NOTE: If the log-likelihood is NaN, that means that the variable is not fitted for that variable (happens with KDE variables with unseen instances)
                log_likelihood += np.nan_to_num(cpd.logl(X), nan=NAN_LOGL_VALUE)

        return log_likelihood

    def slogl(self, X: pd.DataFrame) -> float:
        """Calculates the sum of the log-likelihood of the model given the data

        Args:
            X (pd.DataFrame): Data

        Returns:
            float: Sum of the log-likelihood of the model given the data
        """
        return self.logl(X).sum()

    def feature_logl(
        self,
        data: pd.DataFrame,
        shared_nodes_list: list[str] = [],
    ) -> pd.DataFrame:
        """Calculates the log-likelihood of each feature in the data

        Args:
            data (pd.DataFrame): The data to calculate the log-likelihood

        Returns:
            pd.DataFrame: The data with the feature log-likelihoods
        """
        out_data = data.copy()
        # We calculate the log-likelihoods of all shared variables
        if shared_nodes_list == []:
            shared_nodes_list = list(set(out_data.columns).intersection(self.nodes()))
        shared_pll_columns = [n + "_pll" for n in shared_nodes_list]

        for node in shared_nodes_list:
            cpd = self.cpd(node)
            out_data[f"{node}_pll"] = cpd.logl(out_data)

        out_data["log_likelihood"] = out_data[shared_pll_columns].sum(axis=1)
        return out_data

    # RFE: score -> score_samples?
    # RFE: bound score between 0 and 1?
    def anomaly_score(self, data: pd.DataFrame) -> np.ndarray:
        """Calculates the unbounded (anomaly) score of the data. It is the negative scaled log-likelihood of the data.
        It is a value between 0 and infinity. The higher it is, the more anomalous the data is.

        score = - log [P(n) - P(M)] = log(P(M)) - log(P(n))

        Args:
            data (pd.DataFrame): The data to calculate the score

        Returns:
            np.ndarray: The data with the scores
        """
        return self.max_logl_df["mrv_ell"].sum() - self.logl(data.astype(float))

    def feature_anomaly_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the unbounded (anomaly) score of each feature in the data
        score = - log [P(n) - P(M)] = log(P(M)) - log(P(n))

        Args:
            data (pd.DataFrame): The data to calculate the scores

        Returns:
            pd.DataFrame: The data with the feature scores
        """
        data = self.feature_logl(data)

        new_columns = {}
        for node in self.nodes():
            max_logl_value = self.max_logl_df.loc[
                self.max_logl_df["mrv"] == node, "mrv_ell"
            ].values[0]
            new_columns[f"{node}_score"] = max_logl_value - data[f"{node}_pll"]

        # Use pd.concat to add all new columns at once
        new_columns_df = pd.DataFrame(new_columns)
        data = pd.concat([data, new_columns_df], axis=1)
        data["anomaly_score"] = data[self.score_columns].sum(axis=1)
        return data

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Predicts the probabilities of the data (scaled by the maximum log-likelihood of the model)

        Args:
            data (pd.DataFrame): The data to predict

        Returns:
            pd.DataFrame: The data with the probabilities
        """
        # if is_scaled:
        #     return np.exp(-self.anomaly_score(data))
        # else:
        return np.exp(self.logl(X))

    def sample(self, sample_size: int, seed: int | None = None) -> pd.DataFrame:
        return super().sample(sample_size, seed, ordered=True).to_pandas()

    def explain(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the Feature anomaly scores of the data using log-likelihood scaling and obtaining the most relevant variable statistics.
        Args:
            data (pd.DataFrame): testing data containing self.nodes() columns

        Returns:
            pd.DataFrame: Dataframe containing the following columns:
                - The original columns
                - Anomaly score variables (<variable>_score)
                - The Most Relevant Variable (MRV) and its anomaly score (mrv_score).
                - The Most Relevant Variable (MRV) probability (mrv_prob).
                - The Most Relevant Variable (MRV) contribution to the anomaly score (mrv_contribution).
                - The expected value (mean) of the MRV (expected_mrv_value).
        """

        # Variable initialization
        data.loc[:, "parent_0"] = 1
        data = self.feature_anomaly_score(data)

        # Explanation metrics
        # Most Relevant Variable (Variable with the highest anomaly score)
        data.loc[:, "mrv"] = (
            data[self.score_columns]
            .idxmax(axis=1)
            .str.replace("_score$", "", regex=True)
        )
        # Auxiliary operations
        data.loc[:, "cpd"] = data["mrv"].apply(self.cpd)
        data.loc[:, "parents"] = data["cpd"].apply(
            lambda x: ["parent_0"] + x.evidence()
        )
        # Most Relevant Variable's Score
        data.loc[:, "mrv_score"] = data[self.score_columns].max(axis=1)
        # Most Relevant Variable's probability (between 0 and 1) explaining how likely this value is for the MRV
        data["mrv_prob"] = np.exp(-data["mrv_score"])
        # Most Relevant Variable's contribution to the total score
        data.loc[:, "mrv_contribution"] = data["mrv_score"] / data["anomaly_score"]

        # Most Relevant Variable's expected value (mean of the Gaussian distribution)
        # The mean can be calculated like the dot product of the parent values and its beta coefficients for each row
        data.loc[:, "expected_mrv_value"] = data.apply(
            self._calculate_expected_mrv_value,
            axis=1,
        )

        # Remove extra columns
        data.drop(columns=["parent_0", "parents", "cpd"], inplace=True)

        return data

    # TODO: Rename to plot
    def show(self, ax: matplotlib.axes.Axes | None = None, file_name: str = "") -> None:
        """Shows the BN structure and parameters.
        If ax is provided, then the plot is shown in the ax.
        If file_name is provided, then the plot is saved in the file_name.

        Args:
            ax (matplotlib.axes, optional): Matplotlib ax. Defaults to None.
            file_name (str, optional): File path. Defaults to None.
        """
        print(f"No. of Nodes: {len(self.nodes())}")
        print(f"No. of Arcs: {self.num_arcs()}")

        # RFE: Print pyagrum plot
        # BN structure plot
        gnb.showBN(
            self.graphic,
        )
        # Store cpd strings for each node
        cpd_list = [str(self.cpd(node)) for node in self.nodes()]
        print("\n".join(cpd_list))

    def save(self, model_file: Path) -> None:
        """Saves the model to a file

        Args:
            model_file (Path): The model file path
        """
        with open(model_file, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.graphic.saveBIFXML(str(model_file.with_suffix(".bifxml")))
        nodeColor = {
            node: NODE_TYPE_COLOR_MAP.get(node_type, 0.0)
            for node, node_type in self.node_types().items()
        }
        gumimage.export(
            self.graphic,
            str(model_file.with_suffix(".pdf")),
            nodeColor=nodeColor,
        )

        # Store cpd strings for each node
        cpd_list = [str(self.cpd(node)) for node in self.nodes()]
        with open(model_file.with_suffix(".txt"), "w") as f:
            f.write("\n".join(cpd_list))

    @classmethod
    def load(cls, model_file: Path) -> "BayesianNetwork":
        """Loads the model from a file

        Args:
            model_file (Path): File path

        Returns:
            BayesianNetwork: The model
        """
        with open(model_file, "rb") as f:
            return pickle.load(f)
        self.graphic.loadBIFXML(str(model_file.with_suffix(".bifxml")))

    def _remove_zero_variance_nodes(
        self,
        X: pd.DataFrame,
    ) -> None:
        """
        Filters out variables (nodes) from the input DataFrame that have zero variance and updates internal lists accordingly.
        Args:
            X (pd.DataFrame): The input DataFrame containing variables as columns.
        Returns:
            list[str]: A list of variable names (nodes) that have non-zero variance.
        Side Effects:
            - Updates self.arc_blacklist to retain only arcs where both nodes have non-zero variance.
            - Updates self.type_whitelist to retain only types associated with nodes that have non-zero variance.
        """
        # Remove variables (nodes) with zero variance from the model
        nodes = X.columns.tolist()
        removable_nodes = set(get_zero_variance_variables(X, nodes))
        if len(removable_nodes) > 0:
            # Update feature names and counts
            self.feature_names_in_ = [
                node for node in nodes if node not in removable_nodes
            ]
            self.n_features_in_ = len(self.feature_names_in_)

            # NOTE: First remove arcs to avoid errors in pybnesian
            # Remove arcs involving removable nodes
            for arc in self.arcs():
                if arc[0] in removable_nodes or arc[1] in removable_nodes:
                    self.remove_arc(arc[0], arc[1])
                    self.graphic.eraseArc(arc[0], arc[1])

            # Remove nodes
            for node in removable_nodes.intersection(self.nodes()):
                self.remove_node(node)
                self.graphic.erase(node)

            # Retain only arcs and types with valid nodes
            valid_nodes = set(self.feature_names_in_)
            self.arc_blacklist = [
                arc
                for arc in self.arc_blacklist
                if arc[0] in valid_nodes and arc[1] in valid_nodes
            ]
            self.type_whitelist = [
                t for t in self.type_whitelist if t[0] in valid_nodes
            ]

    # NOTE: Override in specific cases
    def _fit_structure(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pbn.BayesianNetwork:
        """
        Fits the structure of a Bayesian Network to the provided data using a hill-climbing algorithm.

        Parameters:
            X (pd.DataFrame): Feature data to fit the Bayesian Network structure.
            y (pd.Series | None, optional): Target variable to include in the structure learning. If provided, it is concatenated to X.

        Returns:
            pbn.BayesianNetwork: The fitted Bayesian Network structure.

        Notes:
            - Various structure learning parameters (e.g., score, operators, arc/type blacklists/whitelists, etc.) are passed to the underlying `pbn.hc` function.
            - If `y` is provided, it is treated as an additional variable in the network.
            - The random seed may not work with certain scoring methods (e.g., SPBNs).
        """
        if y is not None:
            # If y is provided, we add it to the data
            data = pd.concat([X, y], axis=1)
        else:
            data = X
        # We fit the Bayesian Network structure
        # # NOTE: Making SPBN structure learning start with CKDEType() for all nodes for structure learning
        # if self.bn_type == pbn.SemiparametricBNType():
        #     self.type_whitelist = [
        #         (node, pbn.CKDEType()) for node in self.feature_names_in_
        #     ] + self.type_whitelist
        bn = pbn.hc(
            df=data,
            # start=self,
            bn_type=self.bn_type,
            score=self.search_score,  # SPBN only work with "cv-lik", "holdout-lik", "validated-lik"
            # operators=self.search_operators,
            operators=["arcs"],
            arc_blacklist=self.arc_blacklist,
            arc_whitelist=self.arc_whitelist,
            type_blacklist=self.type_blacklist,
            type_whitelist=self.type_whitelist,
            callback=self.callback,
            max_indegree=self.max_indegree,
            max_iters=self.max_iters,
            epsilon=self.epsilon,
            patience=self.patience,
            seed=self.seed,  # ! This doesn't work with SPBNs (also when score=None)
            num_folds=self.num_folds,
            test_holdout_ratio=self.test_holdout_ratio,
            verbose=self.verbose,
        )

        return bn

    def _fit_node_types(
        self,
        bn: pbn.BayesianNetwork,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pbn.BayesianNetwork:
        if "node_type" in self.search_operators:
            if y is not None:
                # If y is provided, we add it to the data
                data = pd.concat([X, y], axis=1)
            else:
                data = X
            bn = pbn.hc(
                df=data,
                start=bn,
                score=self.search_score,
                operators=["node_type"],
                # arc_blacklist=self.arc_blacklist,
                # arc_whitelist=self.arc_whitelist,
                # type_blacklist=self.type_blacklist,
                # type_whitelist=self.type_whitelist,
                callback=self.callback,
                max_iters=self.max_iters,
                epsilon=self.epsilon,
                patience=self.patience,
                seed=self.seed,
                num_folds=self.num_folds,
                test_holdout_ratio=self.test_holdout_ratio,
                verbose=self.verbose,
            )
        return bn

    def _copy_bn_structure(
        self,
        arcs: list[tuple[str, str]] = [],
        node_types: list[tuple[str, pbn.FactorType]] = [],
        node_num_categories_dict: dict[str, int] = {},
    ) -> None:
        """Copies the structure of a Bayesian Network to the current Bayesian Network.
        Must be called in __setstate__ to update the structure of the Bayesian Network and after the structure learning process.
        """

        for node, new_type in node_types:
            if not self.contains_node(node):
                self.add_node(node)
                if node not in self.graphic.names():  # type: ignore library
                    num_categories = node_num_categories_dict.get(node, 2)
                    self.graphic.add(node, num_categories)
            self.set_node_type(node, new_type)

        for source, target in arcs:
            if not self.has_arc(source, target) and self.can_add_arc(source, target):
                self.add_arc(source, target)
                if not self.graphic.existsArc(source, target):
                    self.graphic.addArc(source, target)

        # NOTE: Important to update the score_columns with the new structure
        self.score_columns = [n + "_score" for n in self.nodes()]

    # RFE: Review the next functions
    def _logl_objective_function(self, x: np.ndarray, node: str) -> float:
        """Objective function to be maximized with the format:
                fun(x, *args) -> float
            This function evaluates the logCPD of a node with an instantiation of its parents

        Args:
            x (numpy.ndarray): 1-D array with shape (n,). We have n float values for each of an instantiation of a BN CPD (n = #node + #parents)
            bn (pybnesian.BayesianNetworkBase): Fitted Bayesian network
            node (str): Node of the CPD to be evaluated with its parents instantiation

        Returns:
            float: Log-likelihood of x given the BN CPD of a node
        """
        cpd = self.cpd(node)
        parents = cpd.evidence()
        columns = [node] + parents

        df = pd.DataFrame([x], columns=columns)
        ll = cpd.logl(df)[0]
        return ll

    # RFE: Add in future versions
    def _calculate_max_logl(self, data: pd.DataFrame) -> None:
        """Calculates the maximum log-likelihood of each node of a Bayesian network fitted from data

        Args:
            data (DataFrame): Data used to fit the Bayesian network and to calculate the maximum log-likelihood of each node

        Returns:
            DataFrame: Table containing each node and its corresponding maximum log-likelihood value.
        """
        nodes = self.nodes()
        mrv_ell = []
        mrv_ev = []

        for node in nodes:
            node_type = self.node_type(node).__str__()
            cpd = self.cpd(node)
            parents = cpd.evidence()

            if node_type == "LinearGaussianFactor":
                # Gaussian evaluated at the mean
                if type(self.cpd(node)) == pbn.CLinearGaussianCPD:
                    # assignment = pbn.Assignment({self.true_label: "0"})
                    # self.cpd(node).conditional_factor(assignment)
                    # assignment = pbn.Assignment({self.true_label: "1"})
                    # self.cpd(node).conditional_factor(assignment)
                    ell = None
                    ev = None
                else:
                    ell = -np.log(np.sqrt(2 * np.pi * self.cpd(node).variance))
                    # ev mean depends on the parents instance values
                    ev = None
            elif node_type == "CKDEFactor":
                bounds = [(data[node].min(), data[node].max())] + [
                    (data[p].min(), data[p].max()) for p in parents
                ]  # bounds of the search for each node and parent
                # stochastic global optimization
                result = differential_evolution(
                    lambda x: -self._logl_objective_function(x, node), bounds=bounds
                )

                ev = result.x[0]
                # We were minimizing -f(x), so we have to change the sign
                ell = -result.fun
            elif node_type == "DiscreteFactor":
                ell = None
                ev = None
            else:
                raise ValueError(f"Unknown node type: {node_type}")
            mrv_ev.append(ev)
            mrv_ell.append(ell)

        self.max_logl_df = pd.DataFrame(
            {
                "mrv": nodes,
                "mrv_ev": mrv_ev,
                "mrv_ell": mrv_ell,
            }
        )

    def _calculate_expected_mrv_value(self, row: pd.Series) -> float:
        """Calculates the expected value of the Most Relevant Variable (MRV) given the parents instantiation

        Args:
            row (pd.Series): Row of the data containing the instantiation of the parents of the MRV

        Returns:
            float: Expected value of the MRV given the parents instantiation
        """
        result = np.nan
        if row["cpd"].type() == pbn.LinearGaussianCPDType():
            result = row[row["parents"]].dot(row["cpd"].beta)
        elif row["cpd"].type() == pbn.CKDEType():
            result = self.max_logl_df.loc[
                self.max_logl_df["mrv"] == row["mrv"], "mrv_ev"
            ].values[0]

        return result
