import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pybnesian as pbn
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

RUTILE_AI_PATH = Path("/app/dev/rutile-ai")
sys.path.append(str(RUTILE_AI_PATH))

from rutile_ai.engine.classification.spbnclassify.src.bnc import (
    GaussianBayesianNetworkAugmentedNaiveBayes,
    GaussianNaiveBayes,
    SemiParametricNaiveBayes,
)
from rutile_ai.engine.classification.spbnclassify.tests.helpers.data import (
    DATA_SIZE,
    SEED,
    TRUE_CLASS_LABEL,
    generate_non_normal_data_classification,
    generate_normal_data_classification,
)


class OracleValidatedScore(pbn.ValidatedScore):
    """
    Oracle score used for testing structure learning.

    It favors the following DAG:
        a -> c <- b
              |
              v
              d
    """

    def __init__(self) -> None:
        super().__init__()
        self.variables: list[str] = ["a", "b", "c", "d"]

    def has_variables(self, vars: list[str]) -> bool:
        """Return whether all given variables belong to the oracle domain."""
        return set(vars).issubset(set(self.variables))

    def compatible_bn(self, model: pbn.BayesianNetworkBase) -> bool:
        """Checks whether the model is compatible (can be used) with this Score."""
        return self.has_variables(model.nodes())

    def local_score(
        self, model: pbn.BayesianNetworkBase, variable: str, evidence: list[str]
    ) -> float:
        """Returns the local score value of a node variable in the model given its parents (evidence).
        Only the version with 3 arguments score.local_score(model, variable, evidence) needs to be implemented. The version with 2 arguments cannot be overriden.
        """
        # Use local decomposition proxy during search: only target local likelihood matters.
        if variable == "c":
            value: float = -1.0
            if "a" in evidence:
                value += 1.0
            if "b" in evidence:
                value += 1.5
            return value
        if variable == "d" and evidence == ["c"]:
            return 1.0
        return -1.0

    # def local_score_node_type(
    #     self,
    #     model: pbn.BayesianNetworkBase,
    #     variable_type: pbn.FactorType,
    #     variable: str,
    #     evidence: list[str],
    # ) -> float:
    #     """Returns the local score value of a node variable in the model if its conditional distribution were a variable_type factor and it had evidence as parents.
    #     This method is optional. This method is only needed if the score is used together with ChangeNodeTypeSet
    #     """
    #     return 0.0

    # def score(self, model: pbn.BayesianNetworkBase) -> float:
    #     """This method is optional. The default implementation sums the local score for all the nodes."""
    #     return sum(
    #         self.local_score(model, node, model.parents(node)) for node in model.nodes()
    #     )

    # def vlocal_score_node_type(
    #     self,
    #     model: pbn.BayesianNetworkBase,
    #     variable_type: pbn.FactorType,
    #     variable: str,
    #     evidence: list[str],
    # ) -> float:
    #     """
    #     Returns the validated local score value of a node variable in the model if its conditional distribution were a variable_type factor and it had evidence as parents.
    #     This method is optional. This method is only needed if the score is used together with ChangeNodeTypeSet.
    #     """
    #     return 0.0

    # def vscore(self, model: pbn.BayesianNetworkBase) -> float:
    #     """Validation score. Default behavior is summing validation local scores.
    #     This method is optional. The default implementation sums the validation local score for all the nodes
    #     """
    #     return sum(
    #         self.vlocal_score(model, node, model.parents(node))
    #         for node in model.nodes()
    #     )

    # def data(self) -> pd.DataFrame:
    #     """Returns the DataFrame used to calculate the score and local scores.
    #     This method is optional.
    #     It is needed to infer the default node types in the GreedyHillClimbing algorithm.
    #     """
    #     return pd.DataFrame(columns=self.variables)

    def vlocal_score(
        self, model: pbn.BayesianNetworkBase, variable: str, evidence: list[str]
    ) -> float:
        """Validation local score with the required 3-argument signature.
        Only the version with 3 arguments score.vlocal_score(model, variable, evidence) needs to be implemented. The version with 2 arguments can not be overriden.
        """
        # This is a simplified version without the validated likelihood proxy: the local score is directly used as validation local score.
        return self.local_score(model, variable, evidence)

    # def vlocal_score_node_type(
    #     self,
    #     model: pbn.BayesianNetworkBase,
    #     variable_type: pbn.FactorType,
    #     variable: str,
    #     evidence: list[str],
    # ) -> float:
    #     """
    #     Returns the validated local score value of a node variable in the model if its conditional distribution were a variable_type factor and it had evidence as parents.
    #     This method is optional. This method is only needed if the score is used together with ChangeNodeTypeSet.
    #     """
    #     return 0.0


class ConditionalLogLikelihoodValidatedScore(pbn.ValidatedScore):
    """Validated score that optimizes conditional log-likelihood for a target variable."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        model_class: type[pbn.BayesianNetworkBase],
        test_ratio: float = 0.2,
        k: int = 10,
        seed: int | None = None,
        construction_args: pbn.Arguments = pbn.Arguments(),
    ) -> None:
        super().__init__()
        self._data = df
        self.target = target

        self.model_class = model_class
        self.feature_names_in_ = df.columns.drop(target).tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        if self.target not in df.columns:
            raise ValueError(f"Target '{target}' is not present in DataFrame columns.")

        # CLL requires enumerating target values to normalize p(y|x).
        self._target_values = (
            pd.Series(df[self.target]).dropna().sort_values().unique().tolist()
        )
        if len(self._target_values) < 2:
            raise ValueError(
                "ConditionalLogLikelihoodScore requires at least two target values."
            )
        # Use stratified holdout split
        stratified_shuffle = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        )
        train_idx, test_idx = next(stratified_shuffle.split(df, df[self.target]))

        self._training_data_holdout = df.iloc[train_idx].reset_index(drop=True)
        self._test_data_holdout = df.iloc[test_idx].reset_index(drop=True)

        # Use stratified K-fold for cross-validation on training data
        self._stratified_kfold = StratifiedKFold(
            n_splits=k, shuffle=True, random_state=seed
        )

    def _get_cv_splits(self) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate stratified K-fold cross-validation splits on the holdout training data."""
        splits = []
        y_train = self._training_data_holdout[self.target]

        for train_idx, test_idx in self._stratified_kfold.split(
            self._training_data_holdout, y_train
        ):
            train_fold = self._training_data_holdout.iloc[train_idx].reset_index(
                drop=True
            )
            test_fold = self._training_data_holdout.iloc[test_idx].reset_index(
                drop=True
            )
            splits.append((train_fold, test_fold))

        return splits

    def has_variables(self, vars: str | list[str]) -> bool:
        """Return whether all given variables belong to the oracle domain."""
        return set(vars).issubset(set(self._data.columns))

    def compatible_bn(self, model: pbn.BayesianNetworkBase) -> bool:
        """Checks whether the model is compatible (can be used) with this Score."""
        return self.has_variables(model.nodes())

    def local_score(
        self, model: pbn.BayesianNetworkBase, variable: str, evidence: list[str]
    ) -> float:
        """Returns the local score value of a node variable in the model given its parents (evidence).
        Match ValidatedLikelihood::local_score behavior: CV over holdout training data.
        Only the version with 3 arguments score.local_score(model, variable, evidence) needs to be implemented. The version with 2 arguments cannot be overriden.
        """
        candidate_model = self._model_with_variable_evidence(model, variable, evidence)

        cll = 0.0
        for train_df, test_df in self._get_cv_splits():
            cll += self._conditional_log_likelihood(candidate_model, train_df, test_df)
        return cll

    def vlocal_score(
        self, model: pbn.BayesianNetworkBase, variable: str, evidence: list[str]
    ) -> float:
        """Validation local score with the required 3-argument signature.
        Match ValidatedLikelihood::vlocal_score behavior: fit on holdout training, score on holdout test.
        Only the version with 3 arguments score.vlocal_score(model, variable, evidence) needs to be implemented. The version with 2 arguments can not be overriden.
        """

        candidate_model = self._model_with_variable_evidence(model, variable, evidence)
        return self._conditional_log_likelihood(
            candidate_model,
            self._training_data_holdout,
            self._test_data_holdout,
        )

    def local_score_node_type(
        self,
        model: pbn.BayesianNetworkBase,
        variable_type: pbn.FactorType,
        variable: str,
        evidence: list[str],
    ) -> float:
        """Return the cross-validated local score for an explicit node type."""
        candidate_model = self._model_with_variable_evidence(model, variable, evidence)
        candidate_model.set_node_type(variable, variable_type)

        cll = 0.0
        for train_df, test_df in self._get_cv_splits():
            cll += self._conditional_log_likelihood(candidate_model, train_df, test_df)
        return cll

    def vlocal_score_node_type(
        self,
        model: pbn.BayesianNetworkBase,
        variable_type: pbn.FactorType,
        variable: str,
        evidence: list[str],
    ) -> float:
        """Return the validation local score for an explicit node type."""
        candidate_model = self._model_with_variable_evidence(model, variable, evidence)
        candidate_model.set_node_type(variable, variable_type)

        return self._conditional_log_likelihood(
            candidate_model,
            self._training_data_holdout,
            self._test_data_holdout,
        )

    def data(self) -> pd.DataFrame:
        """Returns the DataFrame used to calculate the score and local scores.
        This method is optional.
        It is needed to infer the default node types in the GreedyHillClimbing algorithm.
        """
        return self._data

    def _conditional_log_likelihood(
        self,
        model: pbn.BayesianNetworkBase,
        fit_df: pd.DataFrame | object,
        eval_df: pd.DataFrame | object,
    ) -> float:
        """
        Calculate the conditional log-likelihood of a Bayesian Network model on evaluation data.

        This method fits the model parameters using training data and then computes the sum of
        conditional log-likelihoods for each class value in the evaluation data.

        Args:
            model (pbn.BayesianNetworkBase): The Bayesian Network model whose parameters will be fitted.
            fit_df (pd.DataFrame | object): Training data used to fit the model parameters.
                Can be a pandas DataFrame or other compatible data structure.
            eval_df (pd.DataFrame | object): Evaluation data used to compute conditional log-likelihoods.
                Can be a pandas DataFrame or other compatible data structure.

        Returns:
            float: The sum of conditional log-likelihoods across all samples in the evaluation data,
                computed per class value.

        Note:
            - Only model parameters are fitted, not the structure.
            - The target column is excluded from features (fit_X and eval_X).
            - Log-likelihoods are computed conditionally for each class value present in eval_df.
            - Classes with no samples in the evaluation data are skipped.
        """
        fit_df_pd = self._to_pandas(fit_df)
        eval_df_pd = self._to_pandas(eval_df)
        fit_X = fit_df_pd.drop(columns=[self.target])
        fit_y = fit_df_pd[self.target]

        # Only fit the parameters, not the structure
        model._fit_parameters(fit_X, fit_y)

        # Exact conditional log-likelihood:
        # log p(y|x) = log p(x,y) - log p(x)
        class_joint_terms = []
        for class_value in self._target_values:
            class_prior = np.log(model.weights_[class_value])
            class_joint_terms.append(
                class_prior
                + model.conditional_logl(eval_df_pd, class_value=class_value)
            )

        log_joint_matrix = np.column_stack(class_joint_terms)
        log_px = logsumexp(log_joint_matrix, axis=1)

        observed_log_joint = np.empty(len(eval_df_pd), dtype=float)
        for class_value in self._target_values:
            conditional_mask = eval_df_pd[self.target] == class_value
            if not conditional_mask.any():
                continue
            observed_log_joint[conditional_mask.to_numpy()] = np.log(
                model.weights_[class_value]
            ) + model.conditional_logl(
                eval_df_pd.loc[conditional_mask], class_value=class_value
            )

        return float((observed_log_joint - log_px).sum())

    def _model_with_variable_evidence(
        self,
        model: pbn.BayesianNetworkBase,
        variable: str,
        evidence: list[str],
    ) -> pbn.BayesianNetworkBase:
        """
        Create a modified copy of a Bayesian Network with adjusted parent set for a specific variable.
        This method creates a new candidate model based on the current model's configuration
        (classes, weights, feature names, etc.) and modifies its structure to match the desired
        parent set for a given variable. The method only copies the graph structure and node types,
        intentionally excluding CPDs (Conditional Probability Distributions) since the candidate
        parent sets may differ from the current model.
        Args:
            model (pbn.BayesianNetworkBase): The source Bayesian Network model to base the
                candidate model upon.
            variable (str): The target variable whose parent set will be modified.
            evidence (list[str]): A list of variable names that should be parents of the target
                variable in the resulting candidate model.
        Returns:
            pbn.BayesianNetworkBase: A new Bayesian Network model with the same configuration
                as the current model but with the graph structure modified so that only the
                variables in `evidence` are parents of the specified `variable`. Arcs are only
                added if they satisfy the model's validity constraints.
        """
        candidate_model = self.model_class(
            feature_names_in_=self.feature_names_in_,
            n_features_in_=self.n_features_in_,
            true_label=self.target,
        )

        # Copy only graph structure and node types. CPDs are intentionally excluded
        # because candidate parent sets can differ from the current model.
        candidate_model._copy_bn_structure(
            arcs=model.arcs(),
            node_types=list(model.node_types().items()),
        )

        current_parents = set(candidate_model.parents(variable))
        desired_parents = set(evidence)
        # Remove arcs from current parents that are not in the desired parents
        for parent in sorted(current_parents - desired_parents):
            candidate_model.remove_arc(parent, variable)
        # Add arcs from desired parents that are not in the current parents, if valid
        for parent in sorted(desired_parents - current_parents):
            if not candidate_model.has_arc(
                parent, variable
            ) and candidate_model.can_add_arc(parent, variable):
                candidate_model.add_arc(parent, variable)
        return candidate_model

    @staticmethod
    def _to_pandas(df: pd.DataFrame | object) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            return df
        if hasattr(df, "to_pandas"):
            return df.to_pandas()
        raise TypeError(
            "Expected pandas DataFrame or pyarrow RecordBatch-compatible object."
        )


class AccuracyScore(pbn.Score):
    """Score that optimizes classification accuracy on a stratified holdout split."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        model_class: type[pbn.BayesianNetworkBase],
        test_ratio: float = 0.2,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._data = df
        self.target = target
        self.model_class = model_class

        if self.target not in df.columns:
            raise ValueError(f"Target '{target}' is not present in DataFrame columns.")

        self.feature_names_in_ = df.columns.drop(target).tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(df, df[self.target]))
        self._training_data_holdout = df.iloc[train_idx].reset_index(drop=True)
        self._test_data_holdout = df.iloc[test_idx].reset_index(drop=True)

    def has_variables(self, vars: str | list[str]) -> bool:
        """Return whether all given variables belong to the score domain."""
        return set(vars).issubset(set(self._data.columns))

    def compatible_bn(self, model: pbn.BayesianNetworkBase) -> bool:
        """Checks whether the model is compatible with this score."""
        return self.has_variables(model.nodes())

    def score(self, model: pbn.BayesianNetworkBase) -> float:
        """Return holdout accuracy for the given model structure."""
        candidate_model = self._model_from_structure(model)
        return self._accuracy(candidate_model)

    def local_score(
        self, model: pbn.BayesianNetworkBase, variable: str, evidence: list[str]
    ) -> float:
        """Return holdout accuracy after setting variable parents to evidence."""
        candidate_model = self._model_with_variable_evidence(model, variable, evidence)
        return self._accuracy(candidate_model)

    def local_score_node_type(
        self,
        model: pbn.BayesianNetworkBase,
        variable_type: pbn.FactorType,
        variable: str,
        evidence: list[str],
    ) -> float:
        """Return holdout accuracy for candidate structure and variable type."""
        candidate_model = self._model_with_variable_evidence(model, variable, evidence)
        candidate_model.set_node_type(variable, variable_type)
        return self._accuracy(candidate_model)

    def data(self) -> pd.DataFrame:
        """Return the DataFrame used by this score."""
        return self._data

    def _accuracy(self, model: pbn.BayesianNetworkBase) -> float:
        train_x = self._training_data_holdout.drop(columns=[self.target])
        train_y = self._training_data_holdout[self.target]
        test_x = self._test_data_holdout.drop(columns=[self.target])
        test_y = self._test_data_holdout[self.target]

        model._fit_parameters(train_x, train_y)
        pred_y = model.predict(test_x)
        return float(accuracy_score(test_y, pred_y))

    def _model_from_structure(
        self,
        model: pbn.BayesianNetworkBase,
    ) -> pbn.BayesianNetworkBase:
        candidate_model = self.model_class(
            feature_names_in_=self.feature_names_in_,
            n_features_in_=self.n_features_in_,
            true_label=self.target,
        )
        candidate_model._copy_bn_structure(
            arcs=model.arcs(),
            node_types=list(model.node_types().items()),
        )
        return candidate_model

    def _model_with_variable_evidence(
        self,
        model: pbn.BayesianNetworkBase,
        variable: str,
        evidence: list[str],
    ) -> pbn.BayesianNetworkBase:
        candidate_model = self._model_from_structure(model)

        current_parents = set(candidate_model.parents(variable))
        desired_parents = set(evidence)

        for parent in sorted(current_parents - desired_parents):
            candidate_model.remove_arc(parent, variable)

        for parent in sorted(desired_parents - current_parents):
            if not candidate_model.has_arc(
                parent, variable
            ) and candidate_model.can_add_arc(parent, variable):
                candidate_model.add_arc(parent, variable)

        return candidate_model


class F1Score(AccuracyScore):
    """Score that optimizes weighted F1-score on a stratified holdout split."""

    def _accuracy(self, model: pbn.BayesianNetworkBase) -> float:
        train_x = self._training_data_holdout.drop(columns=[self.target])
        train_y = self._training_data_holdout[self.target]
        test_x = self._test_data_holdout.drop(columns=[self.target])
        test_y = self._test_data_holdout[self.target]

        model._fit_parameters(train_x, train_y)
        pred_y = model.predict(test_x)
        return float(f1_score(test_y, pred_y, average="weighted"))


class AUCScore(AccuracyScore):
    """Score that optimizes ROC-AUC on a stratified holdout split."""

    @staticmethod
    def _compute_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
        if y_proba.ndim == 1:
            return float(roc_auc_score(y_true, y_proba))

        if y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))

        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        )

    def _accuracy(self, model: pbn.BayesianNetworkBase) -> float:
        train_x = self._training_data_holdout.drop(columns=[self.target])
        train_y = self._training_data_holdout[self.target]
        test_x = self._test_data_holdout.drop(columns=[self.target])
        test_y = self._test_data_holdout[self.target]

        model._fit_parameters(train_x, train_y)
        pred_proba = model.predict_proba(test_x)
        return self._compute_auc(test_y, pred_proba)


if __name__ == "__main__":

    def _safe_predict_proba(
        model: pbn.BayesianNetworkBase, x: pd.DataFrame
    ) -> np.ndarray | None:
        try:
            return model.predict_proba(x)
        except Exception:
            return None

    def _compute_auc(y_true: pd.Series, y_proba: np.ndarray | None) -> float:
        if y_proba is None:
            return float("nan")

        if y_proba.ndim == 1:
            return float(roc_auc_score(y_true, y_proba))

        if y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))

        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        )

    # CLL score-based structure learning on the toy data.
    df = generate_normal_data_classification(DATA_SIZE, seed=SEED)
    X = df.drop(columns=[TRUE_CLASS_LABEL])
    y = df[TRUE_CLASS_LABEL]

    model_class = GaussianNaiveBayes
    baseline_model_class = GaussianBayesianNetworkAugmentedNaiveBayes

    # model_class = GaussianBayesianNetworkAugmentedNaiveBayes
    cll_score = ConditionalLogLikelihoodValidatedScore(
        df,
        target=TRUE_CLASS_LABEL,
        test_ratio=0.2,
        k=2,
        seed=SEED,
        model_class=model_class,
    )

    df_train = cll_score._training_data_holdout
    df_test = cll_score._test_data_holdout

    X_train = df_train.drop(columns=[TRUE_CLASS_LABEL])
    y_train = df_train[TRUE_CLASS_LABEL]
    X_test = df_test.drop(columns=[TRUE_CLASS_LABEL])
    y_test = df_test[TRUE_CLASS_LABEL]

    # Baselines
    base_model = model_class(seed=42)
    base_model.fit(X_train, y_train)

    baseline_model = baseline_model_class(seed=SEED)
    baseline_model.fit(X_train, y_train)

    # Structure learning with CLL score
    hc = pbn.GreedyHillClimbing()
    cll_pbn = hc.estimate(
        operators=pbn.ArcOperatorSet(), score=cll_score, start=base_model, verbose=True
    )

    cll_pbn.fit(df_train)
    cll_model = model_class(
        feature_names_in_=base_model.feature_names_in_,
        n_features_in_=base_model.n_features_in_,
        classes_=base_model.classes_,
        weights_=base_model.weights_,
        seed=SEED,
    )
    cll_model.copy_pbn(cll_pbn)

    # Structure learning with accuracy score
    acc_score = AccuracyScore(
        df,
        target=TRUE_CLASS_LABEL,
        test_ratio=0.2,
        seed=SEED,
        model_class=model_class,
    )
    acc_pbn = hc.estimate(
        operators=pbn.ArcOperatorSet(), score=acc_score, start=base_model, verbose=True
    )
    acc_pbn.fit(df_train)
    acc_model = model_class(
        feature_names_in_=base_model.feature_names_in_,
        n_features_in_=base_model.n_features_in_,
        classes_=base_model.classes_,
        weights_=base_model.weights_,
        seed=SEED,
    )
    acc_model.copy_pbn(acc_pbn)

    # Structure learning with F1 score
    f1_score_obj = F1Score(
        df,
        target=TRUE_CLASS_LABEL,
        test_ratio=0.2,
        seed=SEED,
        model_class=model_class,
    )
    f1_pbn = hc.estimate(
        operators=pbn.ArcOperatorSet(),
        score=f1_score_obj,
        start=base_model,
        verbose=True,
    )
    f1_pbn.fit(df_train)
    f1_model = model_class(
        feature_names_in_=base_model.feature_names_in_,
        n_features_in_=base_model.n_features_in_,
        classes_=base_model.classes_,
        weights_=base_model.weights_,
        seed=SEED,
    )
    f1_model.copy_pbn(f1_pbn)

    # Structure learning with AUC score
    auc_score_obj = AUCScore(
        df,
        target=TRUE_CLASS_LABEL,
        test_ratio=0.2,
        seed=SEED,
        model_class=model_class,
    )
    auc_pbn = hc.estimate(
        operators=pbn.ArcOperatorSet(),
        score=auc_score_obj,
        start=base_model,
        verbose=True,
    )
    auc_pbn.fit(df_train)
    auc_model = model_class(
        feature_names_in_=base_model.feature_names_in_,
        n_features_in_=base_model.n_features_in_,
        classes_=base_model.classes_,
        weights_=base_model.weights_,
        seed=SEED,
    )
    auc_model.copy_pbn(auc_pbn)

    base_model_pred = base_model.predict(X_test)
    baseline_model_pred = baseline_model.predict(X_test)
    cll_model_pred = cll_model.predict(X_test)
    acc_model_pred = acc_model.predict(X_test)
    f1_model_pred = f1_model.predict(X_test)
    auc_model_pred = auc_model.predict(X_test)

    base_model_proba = _safe_predict_proba(base_model, X_test)
    baseline_model_proba = _safe_predict_proba(baseline_model, X_test)
    cll_model_proba = _safe_predict_proba(cll_model, X_test)
    acc_model_proba = _safe_predict_proba(acc_model, X_test)
    f1_model_proba = _safe_predict_proba(f1_model, X_test)
    auc_model_proba = _safe_predict_proba(auc_model, X_test)

    base_accuracy = accuracy_score(y_test, base_model_pred)
    baseline_accuracy = accuracy_score(y_test, baseline_model_pred)
    cll_accuracy = accuracy_score(y_test, cll_model_pred)
    acc_accuracy = accuracy_score(y_test, acc_model_pred)
    f1_accuracy = accuracy_score(y_test, f1_model_pred)
    auc_accuracy = accuracy_score(y_test, auc_model_pred)

    base_f1 = f1_score(y_test, base_model_pred, average="weighted")
    baseline_f1 = f1_score(y_test, baseline_model_pred, average="weighted")
    cll_f1 = f1_score(y_test, cll_model_pred, average="weighted")
    acc_f1 = f1_score(y_test, acc_model_pred, average="weighted")
    f1_model_f1 = f1_score(y_test, f1_model_pred, average="weighted")
    auc_model_f1 = f1_score(y_test, auc_model_pred, average="weighted")

    base_auc = _compute_auc(y_test, base_model_proba)
    baseline_auc = _compute_auc(y_test, baseline_model_proba)
    cll_auc = _compute_auc(y_test, cll_model_proba)
    acc_auc = _compute_auc(y_test, acc_model_proba)
    f1_auc = _compute_auc(y_test, f1_model_proba)
    auc_model_auc = _compute_auc(y_test, auc_model_proba)

    comparison_rows = [
        {
            "Model": "Base",
            "Arcs": str(sorted(base_model.arcs())),
            "Log-likelihood": base_model.slogl(df),
            "Accuracy": base_accuracy,
            "F1-score (weighted)": base_f1,
            "ROC AUC": base_auc,
        },
        {
            "Model": "Baseline",
            "Arcs": str(sorted(baseline_model.arcs())),
            "Log-likelihood": baseline_model.slogl(df),
            "Accuracy": baseline_accuracy,
            "F1-score (weighted)": baseline_f1,
            "ROC AUC": baseline_auc,
        },
        {
            "Model": "CLLScore",
            "Arcs": str(sorted(cll_model.arcs())),
            "Log-likelihood": cll_model.slogl(df),
            "Accuracy": cll_accuracy,
            "F1-score (weighted)": cll_f1,
            "ROC AUC": cll_auc,
        },
        {
            "Model": "AccuracyScore",
            "Arcs": str(sorted(acc_model.arcs())),
            "Log-likelihood": acc_model.slogl(df),
            "Accuracy": acc_accuracy,
            "F1-score (weighted)": acc_f1,
            "ROC AUC": acc_auc,
        },
        {
            "Model": "F1Score",
            "Arcs": str(sorted(f1_model.arcs())),
            "Log-likelihood": f1_model.slogl(df),
            "Accuracy": f1_accuracy,
            "F1-score (weighted)": f1_model_f1,
            "ROC AUC": f1_auc,
        },
        {
            "Model": "AUCScore",
            "Arcs": str(sorted(auc_model.arcs())),
            "Log-likelihood": auc_model.slogl(df),
            "Accuracy": auc_accuracy,
            "F1-score (weighted)": auc_model_f1,
            "ROC AUC": auc_model_auc,
        },
    ]

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["Log-likelihood"] = comparison_df["Log-likelihood"].map(
        lambda v: f"{v:.6f}"
    )
    comparison_df["Accuracy"] = comparison_df["Accuracy"].map(lambda v: f"{v:.4f}")
    comparison_df["F1-score (weighted)"] = comparison_df["F1-score (weighted)"].map(
        lambda v: f"{v:.4f}"
    )
    comparison_df["ROC AUC"] = comparison_df["ROC AUC"].map(lambda v: f"{v:.4f}")

    print("\nModel comparison table:\n")
    print(comparison_df.to_string(index=False))
