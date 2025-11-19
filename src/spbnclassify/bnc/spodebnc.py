import pandas as pd
import pybnesian as pbn
from rutile_ai.data_handler import TRUE_CLASS_LABEL

from .banc import (
    BaseBayesianNetworkClassifier,
    GaussianBayesianNetworkAugmentedNaiveBayes,
    KDEBayesianNetworkAugmentedNaiveBayes,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
)


class SuperParentOneDependenceEstimator(BaseBayesianNetworkClassifier):
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
        super_parent_: str = "",
    ) -> None:

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
        self.super_parent_ = super_parent_

    def __str__(self) -> str:
        """Returns the string representation of the Bayesian Network Classifier

        Returns:
            str: The string representation
        """
        return "Superparent-one-dependence " + super().__str__()

    def _fit_structure(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pbn.BayesianNetwork:
        if y is None:
            raise ValueError("y must be set")
        # Error control for when super_parent is not correctly set
        if self.super_parent_ == "" or self.super_parent_ not in self.feature_names_in_:
            # NOTE: By default, use the first feature as super_parent
            self.super_parent_ = self.feature_names_in_[0]
        super_children = X.drop(columns=[self.super_parent_]).columns
        for node in super_children:
            self.add_arc(self.super_parent_, node)
        return self


class GaussianSuperParentOneDependenceEstimator(
    SuperParentOneDependenceEstimator,
    GaussianBayesianNetworkAugmentedNaiveBayes,
):
    pass


class SemiParametricSuperParentOneDependenceEstimator(
    SuperParentOneDependenceEstimator,
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
):
    pass


class KDESuperParentOneDependenceEstimator(
    SuperParentOneDependenceEstimator,
    KDEBayesianNetworkAugmentedNaiveBayes,
):
    pass
