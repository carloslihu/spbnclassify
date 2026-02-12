import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pycytominer.operations.variance_threshold import variance_threshold
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


def get_zero_variance_variables(
    X: pd.DataFrame, input_variables: list[str], threshold: float = 0.0
):
    """Feature selector that returns all low-variance features

    Args:
        X (pd.DataFrame): Dataframe containing input_variables
        input_variables (list): list of input variables
        threshold (float, optional): Features with a training-set variance lower than this threshold will be removed. Defaults to 0.0 (i.e. remove the features that have the same value in all samples).

    Returns:
        list: list with all the features that have a variance lower than the threshold
    """
    removable_variables = []
    numeric_variables = X[input_variables].select_dtypes(include=float).columns.tolist()

    if len(numeric_variables) > 0 and X.shape[0] > 1:
        vt = VarianceThreshold(threshold=threshold)
        try:
            vt.fit(X[numeric_variables])  # Removes all features with 0 variance
        except ValueError:
            print("All numeric features have 0 variance, so they will all be removed.")
            return numeric_variables

        output_mask = vt.get_support()
        removable_variables = list(itertools.compress(numeric_variables, ~output_mask))
    return removable_variables


def get_low_variability_variables(
    X: pd.DataFrame, input_variables: list[str], threshold: float = 0.95
):
    """Feature selector that returns all low-variability features

    Args:
        X (pd.DataFrame): Dataframe containing input_variables
        input_variables (list): list of input variables
        threshold (float, optional): Features where the most common value has a frequency higher than this threshold will be removed. Defaults to 0.95 (i.e. remove the features that have the same value in 95% of the samples).

    Returns:
        list: list with all the features that have low variability
    """
    removable_variables = []
    for var in input_variables:
        if X[var].value_counts(normalize=True).iloc[0] > threshold:
            removable_variables.append(var)
    return removable_variables


def get_near_zero_variance_variables(
    X: pd.DataFrame,
    input_variables: list[str],
    freq_cut: float = 0.05,
    unique_cut: float = 0.1,
) -> list[str]:
    """Feature selector that returns all near-zero variance features. In other words, features where:
    - The ratio of the frequency of the second most common value to the frequency of the first most common value is below a certain threshold (freq_cut).
    OR
    - The ratio of the number of unique values to the number of samples is below a certain threshold (unique_cut).

    Args:
        X (pd.DataFrame): Dataframe containing input_variables
        input_variables (list[str]): list of input variables
        freq_cut (float, optional): Ratio (2nd most common feature val / most common).
        Must range between 0 and 1. Remove features lower than freq_cut.
        A low freq_cut will remove features that have large difference between
        the most common feature value and second most common feature value.
        (e.g. this will remove a feature: [1, 1, 1, 1, 0.01, 0.01, ...]).
        Defaults to 0.1.
        unique_cut (float, optional): Ratio (num unique features / num samples).
        Must range between 0 and 1.
        Remove features less than unique cut. A low unique_cut will remove features
        that have very few different measurements compared to the number of samples.
        Defaults to 0.1.

    Returns:
        list[str]: list with all the features that have near-zero variance
    """

    removable_variables = variance_threshold(
        X,
        features=input_variables,  # type: ignore library issue
        freq_cut=freq_cut,
        unique_cut=unique_cut,
    )
    return removable_variables


def correlation_selection(X: pd.DataFrame, threshold: float = 0.7, show: bool = False):
    """Removes from a dataframe the features that have a correlation higher than the threshold.
    IDEA: In theory this should be correlated with the target variable.

    Args:
        X (pd.DataFrame): The dataframe with the input variables.
        threshold(float, optional): The threshold for the correlation matrix. Defaults to 0.7 (i.e. remove the features that have a correlation higher than 0.7).
        show (bool, optional): If True, shows the correlation matrix heatmap. Defaults to False.

    Returns:
        pd.DataFrame: The dataframe without the features that have a correlation higher than the threshold.
    """

    # Selects the columns with more than 1 unique value
    X_res = X[[col for col in X if X[col].nunique() > 1]]
    # Calculates the correlation matrix with absolute values
    corr = X_res.corr(numeric_only=True)
    corr = corr.abs()

    # Shows the correlation matrix hearmap
    if show:
        f, ax = plt.subplots(figsize=(15, 12))
        sns.heatmap(corr, ax=ax)
        plt.show()

    # Gets the upper triangle of the correlation matrix
    corr_triu = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
    corr_triu = corr_triu.stack()

    corr_triu.name = "correlation"
    corr_triu.index.names = ["col1", "col2"]
    corr_df = corr_triu.reset_index()

    # Removes the features that have a correlation higher than the threshold
    redundant_columns = (
        corr_df[corr_df["correlation"] > threshold]
        .index.get_level_values("col2")
        .unique()
    )
    X.drop(redundant_columns, axis=1, inplace=True)
    return X


def feature_importance_selection(X: pd.DataFrame, clf: dict, threshold=0.05):
    """Selects the features of a classifier that have an importance above a threshold.

    Args:
        X (pd.DataFrame): The dataframe with the input variables.
        clf (sklearn classifier): The classifier with a feature_importances_ attribute.
        threshold (float, optional): The feature importance threshold. Defaults to 0.05.

    Returns:
        list: The list of features with an importance above the threshold.
    """

    feat_list = list(zip(X.columns, clf["classifier"].feature_importances_))
    df_imp = pd.DataFrame(feat_list, columns=["feature", "importance"]).sort_values(
        by="importance", ascending=False
    )
    included_feats = df_imp.loc[df_imp["importance"] > threshold, "feature"].values
    return included_feats


def recursive_feature_elimination(
    X: pd.DataFrame,
    y: pd.Series,
    clf: BaseEstimator = DecisionTreeClassifier(),
    show: bool = False,
):
    # Calculates the best set of features for a classifier and returns feature selector
    # IDEA review if this should go before resampling
    # It depends, because resampling is important for class balance
    # But selecting features might depend on the class balance (it usually uses a model to evaluate the selection)
    min_features_to_select = 1  # Minimum number of features to consider
    cv = StratifiedKFold(5)
    # scoring = "f1_macro"
    scoring = "f1_macro"

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )
    rfecv.fit(X, y)

    if show:
        print(f"Optimal number of features: {rfecv.n_features_}")
        n_scores = len(rfecv.cv_results_["mean_test_score"])
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel(f"Mean test {scoring}")
        plt.errorbar(
            range(min_features_to_select, n_scores + min_features_to_select),
            rfecv.cv_results_["mean_test_score"],
            yerr=rfecv.cv_results_["std_test_score"],
        )
        plt.title("Recursive Feature Elimination \nwith correlated features")
        plt.show()
    return rfecv
