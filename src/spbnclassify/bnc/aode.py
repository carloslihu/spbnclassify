import numpy as np
import pandas as pd
import pybnesian as pbn

from .base import BaseMultiBayesianNetworkClassifier
from .spodebnc import (
    GaussianSuperParentOneDependenceEstimator,
    KDESuperParentOneDependenceEstimator,
    SemiParametricSuperParentOneDependenceEstimator,
    SuperParentOneDependenceEstimator,
)


class AveragedOneDependenceEstimator(BaseMultiBayesianNetworkClassifier):
    bn_class = SuperParentOneDependenceEstimator

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network Classifier

        Returns:
            str: The string representation
        """
        return "Averaged one-dependence " + super().__str__()

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "AveragedOneDependenceEstimator":
        """
        Fits the Averaged One-Dependence Bayesian Network Classifier to the provided data.
        This method trains a separate Bayesian network for each feature in the input data,
        treating each as a "super parent" in turn. For each super parent, a Bayesian network
        is instantiated with the specified configuration and trained on the input data.
        Parameters
        ----------
        X : pd.DataFrame
            The input features to train the classifier on.
        y : pd.Series or None, optional
            The target labels corresponding to the input features. If None, unsupervised fitting may be performed.
        Returns
        -------
        AveragedOneDependenceEstimator
            Returns self, the fitted classifier instance.
        """
        super().fit(X, y)

        for super_parent_ in self.feature_names_in_:
            self.bn_dict_[super_parent_] = self.bn_class(
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
                super_parent_=super_parent_,
            )
            self.bn_dict_[super_parent_].fit(X, y)
        return self

    # NOTE: In the case of the averaged one-dependence classifier, the methods are best suited to be learnt starting with the predict_proba method as the base
    def predict_proba(self, X: pd.DataFrame):
        """
        Predict class probabilities for the input samples X using the AODE (Averaged One-Dependence Estimators) model.
        This method computes the posterior probabilities for each class by averaging the probabilities predicted by
        each Bayesian network (one per feature as super-parent) in the ensemble.
        Parameters
        ----------
        X : pd.DataFrame
            Input samples with shape (n_samples, n_features). Each row corresponds to a sample, and each column
            corresponds to a feature. The columns must match the feature names used during model training.
        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) containing the predicted probabilities for each class.
            Each row sums to 1 and corresponds to the probability distribution over classes for a sample.
        """

        result_df = X.copy()
        class_posteriori_probabilities = [f"{k}_score" for k in self.classes_]
        result_df[class_posteriori_probabilities] = 0.0
        for super_parent_ in self.feature_names_in_:
            result_df[class_posteriori_probabilities] += self.bn_dict_[
                super_parent_
            ].predict_proba(X)
        result_df[class_posteriori_probabilities] /= self.n_features_in_

        return result_df[class_posteriori_probabilities].to_numpy()

    def logl(self, X: pd.DataFrame) -> np.ndarray:
        return np.log(self.predict_proba(X))

    # # Unnecessary for the averaged one-dependence classifier
    # def conditional_logl(self, data: pd.DataFrame, class_value: str) -> np.ndarray:
    #     pass

    def sample(self, sample_size: int, seed: int | None = None) -> pd.DataFrame:
        """
        Generate a sample DataFrame by drawing samples uniformly from each super parent feature's Bayesian network.
        Parameters:
            sample_size (int): Total number of samples to generate.
            seed (int | None, optional): Random seed for reproducibility.
        Returns:
            pd.DataFrame: A DataFrame containing the generated samples, shuffled and concatenated from each super parent feature.
        Notes:
            - The total sample size is distributed as evenly as possible among all super parent features.
            - Any remainder from the division is added to the last feature to ensure the total number of samples matches `sample_size`.
            - The resulting DataFrame is shuffled using the provided seed.
        """

        # Calculate the number of samples for each super parent feature with uniform distribution
        sp_sample_sizes = [int(sample_size / self.n_features_in_)] * self.n_features_in_
        # Distribute the remaining samples to ensure the sum is equal to sample_size
        if sum(sp_sample_sizes) != sample_size:
            sp_sample_sizes[-1] += sample_size - sum(sp_sample_sizes)

        # Generate samples for each super parent feature
        sp_sample_list = []
        for sp, sp_sample_size in zip(self.feature_names_in_, sp_sample_sizes):
            sp_sample_df = self.bn_dict_[sp].sample(sp_sample_size, seed)
            sp_sample_list.append(sp_sample_df)
        # Concatenate and shuffle the samples
        sample_df = pd.concat(sp_sample_list, ignore_index=True)
        sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return sample_df

    # TODO: compare_bn_distribution(


class GaussianAveragedOneDependenceEstimator(AveragedOneDependenceEstimator):
    bn_type = pbn.CLGNetworkType()
    bn_class = GaussianSuperParentOneDependenceEstimator


class SemiParametricAveragedOneDependenceEstimator(AveragedOneDependenceEstimator):
    bn_type = pbn.SemiparametricBNType()
    bn_class = SemiParametricSuperParentOneDependenceEstimator


class KDEAveragedOneDependenceEstimator(AveragedOneDependenceEstimator):
    bn_type = pbn.SemiparametricBNType()  # NOTE: To allow Hybrid Bayesian Networks
    bn_class = KDESuperParentOneDependenceEstimator
