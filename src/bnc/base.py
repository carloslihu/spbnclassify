import pickle
from abc import abstractmethod
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pdfkit
import pyagrum as gum
import pyagrum.lib.bn_vs_bn as gcm
import pyagrum.lib.notebook as gnb
import pybnesian as pbn

from ..bn import (
    BayesianNetwork,
    hamming_distance,
    node_presence_distance,
    node_type_distance,
)
from ..utils import (
    NAN_LOGL_VALUE,
    TRUE_CLASS_LABEL,
    dict2html,
    extract_class_name,
    safe_exp,
)


class BaseBayesianNetworkClassifier(BayesianNetwork):
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
        true_label: str = TRUE_CLASS_LABEL,
        prediction_label: str = "predicted_label",
        classes_: list[str] = [],
        weights_: dict[str, float] = {},
    ):
        super().__init__(
            search_score=search_score,
            arc_blacklist=arc_blacklist,
            arc_whitelist=arc_whitelist,
            type_blacklist=type_blacklist,
            type_whitelist=type_whitelist,
            callback=callback,
            max_indegree=max_indegree,
            max_iters=max_iters,
            epsilon=epsilon,
            patience=patience,
            seed=seed,
            num_folds=num_folds,
            test_holdout_ratio=test_holdout_ratio,
            max_train_data_size=max_train_data_size,
            verbose=verbose,
            feature_names_in_=feature_names_in_,
            n_features_in_=n_features_in_,
            true_label=true_label,
            prediction_label=prediction_label,
        )
        self.classes_ = sorted(classes_)
        self.weights_ = weights_

    # TODO: Make this automatically insert the structure or change name
    def _init_structure(
        self,
        nodes: list[str] = [],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, pbn.FactorType]]]:
        """
        Initializes the structure of a Bayesian network classifier.
        This method generates the arcs (directed edges) and node types for the Bayesian network
        based on the provided list of nodes. The arcs represent the relationships between the
        true label and the other nodes, while the node types define the type of each node in
        the network.
        Args:
            nodes (list[str]): A list of node names to include in the Bayesian network.
                The `true_label` node is automatically included and connected to other nodes.
        Returns:
            tuple[list[tuple[str, str]], list[tuple[str, pbn.FactorType]]]:
                - A list of arcs, where each arc is represented as a tuple of two node names
                (parent, child).
                - A list of node types, where each node type is represented as a tuple of a
                node name and its corresponding `pbn.FactorType`.
        """

        arcs = [(self.true_label, node) for node in nodes if node != self.true_label]

        node_types = [(self.true_label, pbn.DiscreteFactorType())] + [
            (node, pbn.LinearGaussianCPDType())
            for node in nodes
            if node != self.true_label
        ]
        return arcs, node_types

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network Classifier

        Returns:
            str: The string representation
        """
        return f"Bayesian Network Classifier with {self.num_nodes()} nodes and {self.num_arcs()} arcs"

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "BaseBayesianNetworkClassifier":
        if y is None:
            raise ValueError("y must be set")
        self.classes_ = sorted(y.unique())
        self.weights_ = y.value_counts(normalize=True)
        super().fit(X, y)
        return self

    def conditional_factor(self, node: str, class_value: str) -> pbn.Factor:
        """
        Computes the conditional factor for a given node and class value.
        This method retrieves the conditional probability distribution (CPD)
        for the specified node. If the CPD is of type `CLinearGaussianCPD`
        or `HCKDE`, it computes the conditional factor based on the provided
        class value. Otherwise, it directly returns the CPD.
        Args:
            node (str): The name of the node for which the conditional factor
                is to be computed.
            class_value (str): The class value used to condition the CPD.
        Returns:
            pbn.Factor: The conditional factor for the given node and class value.
        """

        cpd = self.cpd(node)
        if isinstance(cpd, (pbn.CLinearGaussianCPD, pbn.HCKDE)):
            assignment = pbn.Assignment({self.true_label: class_value})
            conditional_cpd = cpd.conditional_factor(assignment)
        # RFE: Think if this is interesting to calculate
        # elif isinstance(cpd, (pbn.DiscreteFactor)):

        else:
            conditional_cpd = cpd
        return conditional_cpd

    def conditional_logl(self, data: pd.DataFrame, class_value: str) -> np.ndarray:
        """
        Compute the conditional log-likelihood for a fixed class value.
        This method calculates the log-likelihood of the data given the specified
        class value, using the conditional probability distributions (CPDs) of the
        features in the Bayesian network.
        Args:
            data (pd.DataFrame): A DataFrame containing the feature data for which
                the conditional log-likelihood is to be computed. Each row represents
                an observation, and each column corresponds to a feature.
            class_value (str): The class value for which the conditional log-likelihood
                is to be computed.
        Returns:
            np.ndarray: A NumPy array containing the computed log-likelihood
        Notes:
            - If a conditional probability distribution (CPD) is `None`, it indicates
              that the corresponding variable is not fitted (e.g., a Conditional Linear
              Gaussian (CLG) with zero variance).
            - If the log-likelihood for a variable is NaN, it is replaced with zero
              using `np.nan_to_num`, which typically occurs when the variable is not
              fitted (e.g., a Conditional Kernel Density Estimation (CKDE) with zero
              variance).
            - NOTE: Only used for predict_proba calculation
        """

        log_likelihood = pd.Series(self.weights_[class_value], index=data.index)
        for node in self.feature_names_in_:
            cpd = self.conditional_factor(node=node, class_value=class_value)
            # NOTE: When the CPD is None, that means that the variable is not fitted (CLG with 0 variance)
            if cpd:
                # NOTE: If the log-likelihood is NaN, that means that the variable is not fitted for that variable (happens with KDE variables with unseen instances)
                # TODO: Define common MACRO
                log_likelihood += np.nan_to_num(cpd.logl(data), nan=NAN_LOGL_VALUE)

        return log_likelihood.to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate the posterior probability for each class given input data using Bayes Theorem.

        This method computes the probability of each class `k` for every sample in `X` by applying Bayes Theorem in log-space
        to improve numerical stability. The calculation uses the following formula:
        p(C = k | X) =

            P(C = k) * P(X | C = k)                         P(C = k)
        --------------------------------    =  ----------------------------------------
        sum_j[ P(C = j) * P(X | C = j) ]       sum_j[ P(C = j) * exp(logl_j - logl_k) ]

        where:
            - P(C = k) is the prior probability (weight) of class k,
            - logl_k is the log-likelihood of the sample under class k,
            - logl_j is the log-likelihood under class j (for all classes j).

            X (pd.DataFrame): Input data containing features required for prediction. The DataFrame should include all columns
            specified in `self.feature_names_in_`.

            np.ndarray: Array of shape (n_samples, n_classes) containing the posterior probabilities for each class.
            Each column corresponds to the probability scores for a class, in the order of `self.classes_`.
        """

        result_df = X.copy()
        class_posteriori_probabilities = [f"{k}_score" for k in self.classes_]
        # We calculate the logl of each class (logl_k)
        for k in self.classes_:
            result_df[f"logl_{k}"] = self.conditional_logl(
                result_df[self.feature_names_in_], k
            )
            # RFE: Replace in future with scaled log-likelihood?
            # result_df[f"logl_{k}"] = -self.bn_dict_[k].anomaly_score(result_df[self.feature_names_in_])

        # We calculate the probability of each class given the data:
        for k in self.classes_:
            # sum_j[ P(C = j) * exp(logl_j - logl_k) ]
            result_df[f"{k}_score"] = 0
            for j in self.classes_:
                result_df[f"{k}_score"] += self.weights_[j] * safe_exp(
                    result_df[f"logl_{j}"] - result_df[f"logl_{k}"]
                )
            #               P(C = k)
            # -----------------------------------------
            # sum_j[ P(C = j) * exp(logl_j - logl_k) ]
            result_df[f"{k}_score"] = self.weights_[k] / result_df[f"{k}_score"]

        return result_df[class_posteriori_probabilities].to_numpy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        class_probabilities = [f"p_{k}" for k in self.classes_]
        class_probabilities_df = pd.DataFrame(
            self.predict_proba(X[self.feature_names_in_]),
            columns=class_probabilities,
        )
        # To predict the most probable class, we do not need to calculate the normalizing factor
        prediction_label = pd.Series(
            class_probabilities_df.idxmax(axis=1).apply(extract_class_name),
            name=self.prediction_label,
        )
        return prediction_label


class BaseMultiBayesianNetworkClassifier(BaseBayesianNetworkClassifier):
    bn_class = None  # To be defined in subclasses

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
        true_label: str = TRUE_CLASS_LABEL,
        prediction_label: str = "predicted_label",
        classes_: list[str] = [],
        weights_: dict[str, float] = {},
    ):
        super().__init__(
            search_score=search_score,
            arc_blacklist=arc_blacklist,
            arc_whitelist=arc_whitelist,
            type_blacklist=type_blacklist,
            type_whitelist=type_whitelist,
            callback=callback,
            max_indegree=max_indegree,
            max_iters=max_iters,
            epsilon=epsilon,
            patience=patience,
            seed=seed,
            num_folds=num_folds,
            test_holdout_ratio=test_holdout_ratio,
            max_train_data_size=max_train_data_size,
            verbose=verbose,
            feature_names_in_=feature_names_in_,
            n_features_in_=n_features_in_,
            true_label=true_label,
            prediction_label=prediction_label,
            classes_=classes_,
            weights_=weights_,
        )
        self.bn_dict_ = {}

    def nodes(self) -> list[str]:
        return self.feature_names_in_ + [self.true_label]

    def arcs(self) -> dict[str, list[tuple[str, str]]]:
        arc_dict = {}
        for k, bn in self.bn_dict_.items():
            arc_dict[k] = bn.arcs()
        return arc_dict

    def node_types(self) -> dict[str, dict[str, pbn.FactorType]]:
        node_types_dict = {}
        for k, bn in self.bn_dict_.items():
            node_types_dict[k] = bn.node_types()
            node_types_dict[k][self.true_label] = pbn.DiscreteFactorType()

        return node_types_dict

    def fitted(self) -> bool:
        """Checks if the model is fitted

        Returns:
            bool: True if the model is fitted, False otherwise
        """
        return all(bn.fitted() for bn in self.bn_dict_.values())

    def slogl(self, X: pd.DataFrame) -> float:
        """Calculates the sum of the log-likelihood of the model given the data

        Args:
            X (pd.DataFrame): Data

        Returns:
            float: Sum of the log-likelihood of the model given the data
        """
        return self.logl(X).sum()

    def save(self, model_file: Path) -> None:
        """Saves the model to a file

        Args:
            model_file (Path): The model file path
        """
        for k, bn in self.bn_dict_.items():
            bn.save(model_file.with_name(f"{model_file.stem}_{k}.pkl"))
        with open(model_file, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, model_file: Path) -> "BaseMultiBayesianNetworkClassifier":
        """Loads the model from a file

        Args:
            model_file (Path): The model file path
        """
        with open(model_file, "rb") as f:
            return pickle.load(f)
        for k, v in self.bn_dict_.items():
            self.bn_dict_[k] = cls.bn_class.load(
                model_file.with_name(f"{model_file.stem}_{k}.pkl")
            )

    def compare_bn(
        self,
        source_class_name: str,
        target_class_name: str,
        save_pdf: bool = False,
        sample_size: int = 1000,
        seed: int | None = None,
    ) -> tuple:
        """
        Compares two Bayesian Networks (BNs) corresponding to the given source and target class names.

        The comparison includes both structural and parametric aspects:
        - Structural comparison: Evaluates the shared nodes and arcs between the two BNs and generates a visual report.
        - Parametric comparison: Computes the distance between the probability distributions of the shared nodes.

        Optionally, a PDF report of the structural comparison can be generated.

        Args:
            source_class_name (str): Name of the source class whose BN will be compared.
            target_class_name (str): Name of the target class whose BN will be compared.
            save_pdf (bool, optional): If True, saves a PDF report of the structural comparison. Defaults to False.
            sample_size (int, optional): Number of samples to use for parametric comparison. Defaults to 1000.
            seed (int | None, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - nd: Node difference metric.
                - sd: Structural difference metric.
                - ntd: Node type difference metric.
                - bn_distance: Parametric (distributional) distance between the BNs.
        """
        # We first compare the different nodes
        # BN1 and BN2 are the Bayesian Networks of the source and target classes
        bn1 = self.bn_dict_[source_class_name]
        bn2 = self.bn_dict_[target_class_name]
        nodes1 = bn1.nodes()
        arcs1 = bn1.arcs()
        nodes2 = bn2.nodes()
        arcs2 = bn2.arcs()

        # We calculate the shared nodes and their corresponding arcs for each class
        shared_nodes = set(nodes1).intersection(set(nodes2))
        shared_nodes_list = list(shared_nodes)
        shared_arcs_source_class_name = [
            arc for arc in arcs1 if arc[0] in shared_nodes and arc[1] in shared_nodes
        ]
        shared_arcs_target_class_name = [
            arc for arc in arcs2 if arc[0] in shared_nodes and arc[1] in shared_nodes
        ]
        # We create the graphical BNs with the shared nodes and arcs
        graph1 = gum.BayesNet()
        graph2 = gum.BayesNet()
        graph1.addVariables(shared_nodes)
        graph2.addVariables(shared_nodes)
        graph1.addArcs(shared_arcs_source_class_name)
        graph2.addArcs(shared_arcs_target_class_name)
        # STRUCTURAL COMPARISON
        nd, sd, ntd, html_content = self._compare_bn_structure(bn1, bn2, graph1, graph2)
        # Generate PDF from HTML string
        if save_pdf:
            pdfkit.from_string(
                html_content,
                f"output/bn_diff_{source_class_name}_{target_class_name}.pdf",
            )
        # PARAMETRIC COMPARISON
        bn_distance = self._compare_bn_distribution(
            source_class_name, target_class_name, shared_nodes_list, sample_size, seed
        )
        return nd, sd, ntd, bn_distance

    def show(self, ax: matplotlib.axes.Axes | None = None, file_name: str = "") -> None:
        """Shows the BN structure and parameters.
        If ax is provided, then the plot is shown in the ax.
        If file_name is provided, then the plot is saved in the file_name.

        Args:
            ax (matplotlib.axes, optional): Matplotlib ax. Defaults to None.
            file_name (str, optional): File path. Defaults to None.
        """
        print("Bayesian Multinet Classifier")
        print("classes: ", self.classes_)
        for k, bn in self.bn_dict_.items():
            print(f"{k} BN")
            bn.show(ax=ax, file_name=file_name)

    def _compare_bn_structure(
        self,
        bn1: BayesianNetwork,
        bn2: BayesianNetwork,
        graph1: gum.BayesNet,
        graph2: gum.BayesNet,
    ) -> tuple:
        """Returns the node distance and arc distance between the Bayesian Networks"""
        # Structural Comparison
        nodes1 = bn1.nodes()
        nodes2 = bn2.nodes()
        shared_nodes_list = list(graph1.names().keys())

        # We compare the shared nodes
        nd = node_presence_distance(nodes1, nodes2)
        # We compare the graphical BNs with the shared nodes and arcs
        sd = hamming_distance(bn1, bn2)
        ntd = node_type_distance(bn1, bn2, shared_nodes_list)

        cmp = gcm.GraphicalBNComparator(graph1, graph2)
        bn_diff_html = gnb.getBNDiff(graph1, graph2, size=10)

        html_content = gnb.getSideBySide(
            # graph1,
            # graph2,
            bn_diff_html,
            dict2html(cmp.scores(), cmp.hamming()),
            cmp.equivalentBNs(),
            captions=[
                # "bn1",
                # "bn2",
                "graphical diff",
                "Scores",
                "equivalent ?",
            ],
            valign="bottom",
        )

        return nd, sd, ntd, html_content

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "BaseMultiBayesianNetworkClassifier":
        """Fit the model to the data

        Args:
            X (pd.DataFrame): The input features
            y (pd.Series): The target labels
            class_label (str, optional): The name of the class label column. Defaults to TRUE_CLASS_LABEL.

        Returns:
            BaseMultiBayesianNetworkClassifier: The fitted classifier
        """
        if y is None:
            raise ValueError("y must be set")
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.classes_ = sorted(y.unique())
        self.weights_ = y.value_counts(normalize=True)
        return self

    @abstractmethod
    def logl(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the log-likelihood of the given input data X.
        Parameters:
            X (pd.DataFrame): Input features for which to compute the log-likelihood.
        Returns:
            np.ndarray: Array of log-likelihood values for each sample in X.
        """
        pass

    @abstractmethod
    def conditional_logl(self, data: pd.DataFrame, class_value: str) -> np.ndarray:
        pass

    @abstractmethod
    def sample(self, sample_size: int, seed: int | None = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def _compare_bn_distribution(
        self,
        source_class_name: str,
        target_class_name: str,
        shared_nodes_list: list,
        sample_size: int = 1000,
        seed: int | None = None,
    ) -> float:
        pass

    def __getstate__(self) -> dict:
        """Extends the pickle dump attributes

        Returns:
            dict: Pickle state
        """
        state = self.__dict__.copy()
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
        self.__dict__.update(state)
