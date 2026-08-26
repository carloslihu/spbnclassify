import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pybnesian as pbn
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


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
        # Use local decomposition proxy during search: only true_label local likelihood matters.
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


# region Experimental structure learning scores for classification tasks
class ConditionalLogLikelihoodValidatedScore(pbn.ValidatedScore):
    """Validated score that optimizes conditional log-likelihood for a true_label variable."""

    def __init__(
        self,
        data: pd.DataFrame,
        true_label: str,
        model_class: type[pbn.BayesianNetworkBase],
        test_holdout_ratio: float = 0.2,
        k: int = 10,
        seed: int | None = None,
        construction_args: pbn.Arguments = pbn.Arguments(),
    ) -> None:
        super().__init__()
        self._data = data
        self.true_label = true_label

        self.model_class = model_class
        # BUG: Multinet: true_label not found in axis"
        self.feature_names_in_ = data.columns.drop(true_label).tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        if self.true_label not in data.columns:
            raise ValueError(
                f"Target '{true_label}' is not present in DataFrame columns."
            )

        # CLL requires enumerating true_label values to normalize p(y|x).
        self._target_values = (
            pd.Series(data[self.true_label]).dropna().sort_values().unique().tolist()
        )
        # Use stratified holdout split
        stratified_shuffle = StratifiedShuffleSplit(
            n_splits=1, test_size=test_holdout_ratio, random_state=seed
        )
        train_idx, test_idx = next(
            stratified_shuffle.split(data, data[self.true_label])
        )

        self._training_data_holdout = data.iloc[train_idx].reset_index(drop=True)
        self._test_data_holdout = data.iloc[test_idx].reset_index(drop=True)

        # Use stratified K-fold for cross-validation on training data
        self._stratified_kfold = StratifiedKFold(
            n_splits=k, shuffle=True, random_state=seed
        )

    def _get_cv_splits(self) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate stratified K-fold cross-validation splits on the holdout training data."""
        splits = []
        y_train = self._training_data_holdout[self.true_label]

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
            - The true_label column is excluded from features (fit_X and eval_X).
            - Log-likelihoods are computed conditionally for each class value present in eval_df.
            - Classes with no samples in the evaluation data are skipped.
        """
        fit_df_pd = self._to_pandas(fit_df)
        eval_df_pd = self._to_pandas(eval_df)
        fit_X = fit_df_pd.drop(columns=[self.true_label])
        fit_y = fit_df_pd[self.true_label]

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
            conditional_mask = eval_df_pd[self.true_label] == class_value
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
            variable (str): The true_label variable whose parent set will be modified.
            evidence (list[str]): A list of variable names that should be parents of the true_label
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
            true_label=self.true_label,
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
    def _to_pandas(data: pd.DataFrame | object) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if hasattr(data, "to_pandas"):
            return data.to_pandas()
        raise TypeError(
            "Expected pandas DataFrame or pyarrow RecordBatch-compatible object."
        )


class AccuracyScore(pbn.Score):
    """Score that optimizes classification accuracy on a stratified holdout split."""

    def __init__(
        self,
        data: pd.DataFrame,
        true_label: str,
        model_class: type[pbn.BayesianNetworkBase],
        test_holdout_ratio: float = 0.2,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._data = data
        self.true_label = true_label
        self.model_class = model_class

        if self.true_label not in data.columns:
            raise ValueError(
                f"Target '{true_label}' is not present in DataFrame columns."
            )
        # BUG: Multinet: true_label not found in axis"
        self.feature_names_in_ = data.columns.drop(true_label).tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_holdout_ratio,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(data, data[self.true_label]))
        self._training_data_holdout = data.iloc[train_idx].reset_index(drop=True)
        self._test_data_holdout = data.iloc[test_idx].reset_index(drop=True)

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
        train_x = self._training_data_holdout.drop(columns=[self.true_label])
        train_y = self._training_data_holdout[self.true_label]
        test_x = self._test_data_holdout.drop(columns=[self.true_label])
        test_y = self._test_data_holdout[self.true_label]

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
            true_label=self.true_label,
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
        train_x = self._training_data_holdout.drop(columns=[self.true_label])
        train_y = self._training_data_holdout[self.true_label]
        test_x = self._test_data_holdout.drop(columns=[self.true_label])
        test_y = self._test_data_holdout[self.true_label]

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
        train_x = self._training_data_holdout.drop(columns=[self.true_label])
        train_y = self._training_data_holdout[self.true_label]
        test_x = self._test_data_holdout.drop(columns=[self.true_label])
        test_y = self._test_data_holdout[self.true_label]

        model._fit_parameters(train_x, train_y)
        pred_proba = model.predict_proba(test_x)
        return self._compute_auc(test_y, pred_proba)


# endregion Experimental structure learning scores for classification tasks
