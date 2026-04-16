import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pybnesian as pbn

RUTILE_AI_PATH = Path("/app/dev/rutile-ai")
sys.path.append(str(RUTILE_AI_PATH))

from rutile_ai.engine.classification.spbnclassify.src.bnc import GaussianNaiveBayes
from rutile_ai.engine.classification.spbnclassify.tests.helpers.data import (
    DATA_SIZE,
    SEED,
    TRUE_CLASS_LABEL,
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
        classes: list[str] = [],
        weights: dict[str, float] = {},
    ) -> None:
        super().__init__()
        self._data = df
        self.target = target
        self.model_class = model_class
        self.feature_names_in_ = df.columns.drop(target).tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.classes_ = classes
        self.weights_ = weights

        self.k = k
        self.seed = seed

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

        self.holdout = pbn.HoldoutLikelihood(
            df,
            test_ratio=test_ratio,
            seed=seed,
            construction_args=construction_args,
        ).holdout
        self.cv = pbn.CVLikelihood(
            self.holdout.training_data(),
            k=k,
            seed=seed,
            construction_args=construction_args,
        ).cv

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
        for train_df, test_df in self.cv:
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
            self.holdout.training_data(),
            self.holdout.test_data(),
        )

    # TODO: local_score_node_type, vlocal_score_node_type

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
        """Returns the conditional log-likelihood of the model on the evaluation data."""
        # Copy the model structure

        fit_df_pd = self._to_pandas(fit_df)
        eval_df_pd = self._to_pandas(eval_df)

        fit_X = fit_df_pd.drop(columns=[self.target])
        fit_y = fit_df_pd[self.target]

        # Only fit the parameters, not the structure
        model._fit_parameters(fit_X, fit_y)

        eval_X = eval_df_pd.drop(columns=[self.target])
        eval_df_pd["logl"] = 0.0
        for class_value in self._target_values:
            conditional_mask = eval_df_pd[self.target] == class_value
            if not conditional_mask.any():
                continue
            eval_df_pd.loc[conditional_mask, "logl"] = model.conditional_logl(
                eval_X.loc[conditional_mask], class_value=class_value
            )

        return float(eval_df_pd["logl"].sum())

    def _model_with_variable_evidence(
        self,
        model: pbn.BayesianNetworkBase,
        variable: str,
        evidence: list[str],
    ) -> pbn.BayesianNetworkBase:
        candidate_model = self.model_class(
            classes_=self.classes_,
            weights_=self.weights_,
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

        for parent in sorted(current_parents - desired_parents):
            candidate_model.remove_arc(parent, variable)

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


# TODO: Metric-based structure learning
# TODO: CLL score-based NTL
if __name__ == "__main__":
    start_model = pbn.GaussianNetwork(["a", "b", "c", "d"])

    hc = pbn.GreedyHillClimbing()
    learned_model = hc.estimate(
        operators=pbn.ArcOperatorSet(),
        score=OracleValidatedScore(),
        start=start_model,
        verbose=True,
    )
    assert set(learned_model.arcs()) == {("a", "c"), ("b", "c"), ("c", "d")}

    # CLL score-based structure learning on the toy data.
    df = generate_normal_data_classification(DATA_SIZE, seed=SEED)
    X = df.drop(columns=[TRUE_CLASS_LABEL])
    y = df[TRUE_CLASS_LABEL]

    model_class = GaussianNaiveBayes

    base_model = model_class(seed=42)
    base_model.fit(X, y)
    classes = base_model.classes_
    weights = base_model.weights_

    cll_score = ConditionalLogLikelihoodValidatedScore(
        df,
        target=TRUE_CLASS_LABEL,
        test_ratio=0.2,
        k=2,
        seed=SEED,
        model_class=model_class,
        classes=classes,
        weights=weights,
    )

    learnt_model = hc.estimate(
        operators=pbn.ArcOperatorSet(), score=cll_score, start=base_model, verbose=True
    )
    print("Base model arcs:", sorted(base_model.arcs()))
    print("Base model log-likelihood:", base_model.slogl(df))
    print("Learned model arcs:", sorted(learnt_model.arcs()))
    print("Train CLL:", cll_score.score(learnt_model))
    print("Validation CLL:", cll_score.vscore(learnt_model))
