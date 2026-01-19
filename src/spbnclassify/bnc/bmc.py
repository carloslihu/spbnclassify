import numpy as np
import pandas as pd
import pybnesian as pbn
from scipy.spatial.distance import jensenshannon
from src.spbnclassify.utils import safe_exp
from src.spbnclassify.utils.constants import TRUE_CLASS_LABEL

from ..bn import (
    BayesianNetwork,
    GaussianBayesianNetwork,
    KDEBayesianNetwork,
    SemiParametricBayesianNetwork,
    gaussian_jensen_shannon_divergence,
)
from .base import BaseMultiBayesianNetworkClassifier


class BayesianMultinet(BaseMultiBayesianNetworkClassifier):
    bn_class = BayesianNetwork

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network

        Returns:
            str: The string representation
        """
        return f"Bayesian Multinet Classifier"

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BayesianMultinet":
        super().fit(X, y)

        data = pd.concat([X, y], axis=1)
        for k in self.classes_:
            self.bn_dict_[k] = self.bn_class(
                search_score=self.search_score,
                arc_blacklist=self.arc_blacklist,
                arc_whitelist=self.arc_whitelist,
                type_blacklist=self.type_blacklist,
                type_whitelist=self.type_whitelist,
                callback=self.callback,
                max_indegree=self.max_indegree,
                max_iters=self.max_iters,
                epsilon=self.epsilon,
                patience=self.patience,
                seed=self.seed,
                num_folds=self.num_folds,
                test_holdout_ratio=self.test_holdout_ratio,
                max_train_data_size=self.max_train_data_size,
                verbose=self.verbose,
                feature_names_in_=self.feature_names_in_,
                n_features_in_=self.n_features_in_,
                true_label=self.true_label,
                prediction_label=self.prediction_label,
            )
            filtered_X = data.loc[data[self.true_label] == k, self.feature_names_in_]
            # Module.log_note(
            #     3,
            #     f"Learning {k} BN with training data shape: {filtered_X.shape}",
            # )
            self.bn_dict_[k].fit(filtered_X)
        return self

    def logl(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculates the log-likelihood of the model given the data.
        This method computes the log-likelihood by marginalizing over the classes:
            logl(X) = log[ sum_j [ P(C_i = j) * P(X | C_i = j) ] ]
        where:
            - P(C_i = j) = self.weights_[j] (the weight of class j)
            - P(X | C_i = j) = np.exp(self.bn_dict_[j].logl(X)) (the probability of instance X given the structure of class j)

        Mathematically:
            LL(Multinet | D) = sum_{i=1}^N log( sum_c [ P(c) * P_{B_c}(x^{(i)}) ] )

        Args:
            X (pd.DataFrame): Data

        Returns:
            pd.Series[float]: Log-likelihood of the model given the data
        """

        probability_sum = np.zeros(X.shape[0])
        log_likelihood = np.zeros(X.shape[0])
        for k in self.classes_:
            probability_sum += self.weights_[k] * safe_exp(self.bn_dict_[k].logl(X))

        log_likelihood = np.log(probability_sum)
        return log_likelihood

    def conditional_logl(self, data: pd.DataFrame, class_value: str) -> np.ndarray:
        log_likelihood = self.bn_dict_[class_value].logl(data[self.feature_names_in_])
        return log_likelihood

    def sample(self, sample_size: int, seed: int | None = None) -> pd.DataFrame:
        """
        Generate a stratified sample of data points from the Bayesian network classifiers according to class weights.
        Args:
            sample_size (int): Total number of samples to generate.
            seed (int | None, optional): Random seed for reproducibility. Defaults to None.
        Returns:
            pd.DataFrame: A DataFrame containing the sampled data, with class proportions matching the learned class weights.
        Notes:
            - The method ensures that the total number of samples equals `sample_size` by distributing any rounding errors.
            - Each class's samples are generated using its corresponding Bayesian network.
            - The resulting DataFrame is shuffled before being returned.
        """

        # Calculate the number of samples for each class based on the weights
        class_sample_sizes = np.floor(sample_size * self.weights_.to_numpy()).astype(
            int
        )
        # Distribute the remaining samples to ensure the sum is equal to sample_size
        remaining_samples = sample_size - class_sample_sizes.sum()
        if remaining_samples > 0:
            for i in np.argsort(-self.weights_.to_numpy())[:remaining_samples]:
                class_sample_sizes[i] += 1

        # Generate samples for each class
        class_sample_list = []
        for class_value, class_sample_size in zip(
            self.weights_.keys(), class_sample_sizes
        ):
            class_sample_df = self.bn_dict_[class_value].sample(class_sample_size, seed)
            class_sample_df[self.true_label] = class_value
            class_sample_list.append(class_sample_df)

        # Concatenate and shuffle the class_sample_list
        sample_df = pd.concat(class_sample_list, ignore_index=True)
        sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return sample_df


class GaussianBayesianMultinet(BayesianMultinet):
    bn_type = pbn.CLGNetworkType()
    bn_class = GaussianBayesianNetwork

    def __init__(
        self,
        search_score: str = "bic",
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
    ) -> None:
        super().__init__(
            search_score=search_score,
            arc_blacklist=arc_blacklist,
            arc_whitelist=arc_whitelist,
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
        self.joint_gaussian_ = {}

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network

        Returns:
            str: The string representation
        """
        return "Gaussian " + super().__str__()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BayesianMultinet":
        super().fit(X, y)
        for k in self.classes_:
            self.joint_gaussian_[k] = self.bn_dict_[k].joint_gaussian_
        return self

    def _compare_bn_distribution(
        self,
        source_class_name: str,
        target_class_name: str,
        shared_nodes_list: list,
        sample_size: int = 1000,
        seed: int | None = None,
    ) -> float:
        bn_distance = gaussian_jensen_shannon_divergence(
            (
                self.joint_gaussian_[source_class_name]["mean"][shared_nodes_list].T,
                self.joint_gaussian_[source_class_name]["cov"].loc[
                    shared_nodes_list, shared_nodes_list
                ],
            ),
            (
                self.joint_gaussian_[target_class_name]["mean"][shared_nodes_list].T,
                self.joint_gaussian_[target_class_name]["cov"].loc[
                    shared_nodes_list, shared_nodes_list
                ],
            ),
        )
        return bn_distance

    def _compare_node_distribution(
        self,
        node: str,
        source_class_name: str,
        target_class_name: str,
    ) -> float:

        bn1 = self.bn_dict_[source_class_name]
        bn2 = self.bn_dict_[target_class_name]
        # We get the node in each BN and its respective parents
        # We compare the CPDs of each node with Jensen Shannon Divergence
        # TODO: Think if joint_gaussian is correctly calculated
        # TODO Check what to do comparing nodes with non-common parents
        bn1_nodes = [node] + bn1.parents(node)
        bn2_nodes = [node] + bn2.parents(node)
        print(bn1_nodes)
        print(bn2_nodes)
        bn_distance = gaussian_jensen_shannon_divergence(
            (
                self.joint_gaussian_[source_class_name]["mean"][bn1_nodes].T,
                self.joint_gaussian_[source_class_name]["cov"].loc[
                    bn1_nodes, bn1_nodes
                ],
            ),
            (
                self.joint_gaussian_[target_class_name]["mean"][bn2_nodes].T,
                self.joint_gaussian_[target_class_name]["cov"].loc[
                    bn2_nodes, bn2_nodes
                ],
            ),
        )
        return bn_distance


class SemiParametricBayesianMultinet(BayesianMultinet):
    bn_type = pbn.SemiparametricBNType()
    bn_class = SemiParametricBayesianNetwork

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network

        Returns:
            str: The string representation
        """
        return "SemiParametric " + super().__str__()

    def _compare_bn_distribution(
        self,
        source_class_name: str,
        target_class_name: str,
        shared_nodes_list: list,
        sample_size: int = 1000,
        seed: int | None = None,
    ) -> float:
        # We sample from the source and target BNs
        source_sample = self.bn_dict_[source_class_name].sample(sample_size, seed)[
            shared_nodes_list
        ]
        target_sample = self.bn_dict_[source_class_name].sample(sample_size, seed)[
            shared_nodes_list
        ]
        # We concatenate the samples
        joint_sample = pd.concat(
            [source_sample, target_sample], axis=0, ignore_index=True
        )

        # We calculate the probabilities of each sample given the BNs
        # If the BNs have the same nodes, we simply calculate the probabilities
        if set(self.bn_dict_[source_class_name].nodes()) == set(
            self.bn_dict_[target_class_name].nodes()
        ):
            source_proba = self.bn_dict_[source_class_name].predict_proba(joint_sample)
            target_proba = self.bn_dict_[target_class_name].predict_proba(joint_sample)
        # If the BNs have different nodes, we calculate the probabilities of the shared nodes
        else:
            source_proba = np.exp(
                self.bn_dict_[source_class_name].feature_logl(
                    source_sample, shared_nodes_list
                )["log_likelihood"]
            )
            target_proba = np.exp(
                self.bn_dict_[target_class_name].feature_logl(
                    target_sample, shared_nodes_list
                )["log_likelihood"]
            )
        # We calculate the Jensen Shannon Divergence
        bn_distance = float(jensenshannon(source_proba, target_proba) ** 2)
        return bn_distance


class KDEBayesianMultinet(SemiParametricBayesianMultinet):
    bn_class = KDEBayesianNetwork

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network

        Returns:
            str: The string representation
        """
        return "KDE " + BayesianMultinet.__str__(self)
