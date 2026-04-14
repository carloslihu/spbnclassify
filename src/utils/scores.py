import numpy as np
import pandas as pd
import pybnesian as pbn


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
        """Check if the network nodes are compatible with this score."""
        return self.has_variables(model.nodes())

    def local_score(
        self, model: pbn.BayesianNetworkBase, variable: str, evidence: list[str]
    ) -> float:
        """Training local score used by the search process."""
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

    def vscore(self, model: pbn.BayesianNetworkBase) -> float:
        """Validation score. Default behavior is summing validation local scores."""
        return sum(
            self.vlocal_score(model, node, model.parents(node))
            for node in model.nodes()
        )

    def vlocal_score(
        self, model: pbn.BayesianNetworkBase, variable: str, evidence: list[str]
    ) -> float:
        """Validation local score with the required 3-argument signature."""
        return self.local_score(model, variable, evidence)

    def vlocal_score_node_type(
        self,
        model: pbn.BayesianNetworkBase,
        variable_type: pbn.FactorType,
        variable: str,
        evidence: list[str],
    ) -> float:
        """Optional override used by ChangeNodeTypeSet-enabled searches."""
        return self.vlocal_score(model, variable, evidence)

    def data(self) -> None:
        """No dataset is required by this synthetic score."""
        return None


class ConditionalLogLikelihoodValidatedScore(pbn.ValidatedScore):
    """Validated score that optimizes conditional log-likelihood for a target variable."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        test_ratio: float = 0.2,
        k: int = 10,
        seed: int | None = None,
        construction_args: pbn.Arguments = pbn.Arguments(),
    ) -> None:
        super().__init__()
        if target not in df.columns:
            raise ValueError(f"Target '{target}' is not present in DataFrame columns.")

        self.target = target
        self.holdout_lik = pbn.HoldoutLikelihood(
            df,
            test_ratio=test_ratio,
            seed=seed,
            construction_args=construction_args,
        )
        self.cv_lik = pbn.CVLikelihood(
            self.holdout_lik.training_data(),
            k=k,
            seed=seed,
            construction_args=construction_args,
        )

        # CLL requires enumerating target values to normalize p(y|x).
        self._target_values = (
            pd.Series(df[target]).dropna().sort_values().unique().tolist()
        )
        if len(self._target_values) < 2:
            raise ValueError(
                "ConditionalLogLikelihoodScore requires at least two target values."
            )

    def has_variables(self, vars: str | list[str]) -> bool:
        return self.cv_lik.has_variables(vars)

    def compatible_bn(self, model: pbn.BayesianNetworkBase) -> bool:
        return self.has_variables(model.nodes())

    def local_score(
        self, model: pbn.BayesianNetworkBase, variable: str, evidence: list[str]
    ) -> float:
        # Use local decomposition proxy during search: only target local likelihood matters.
        if variable != self.target:
            return 0.0
        return self.cv_lik.local_score(model, variable, evidence)

    def score(self, model: pbn.BayesianNetworkBase) -> float:
        return self._conditional_log_likelihood(model, self.holdout_lik.training_data())

    def vscore(self, model: pbn.BayesianNetworkBase) -> float:
        return self._conditional_log_likelihood(model, self.holdout_lik.test_data())

    def vlocal_score(
        self, model: pbn.BayesianNetworkBase, variable: str, evidence: list[str]
    ) -> float:
        if variable != self.target:
            return 0.0
        return self.holdout_lik.local_score(model, variable, evidence)

    def vlocal_score_node_type(
        self,
        model: pbn.BayesianNetworkBase,
        variable_type: pbn.FactorType,
        variable: str,
        evidence: list[str],
    ) -> float:
        if variable != self.target:
            return 0.0
        return self.holdout_lik.local_score(model, variable_type, variable, evidence)

    def training_data(self) -> pd.DataFrame:
        return self.holdout_lik.training_data()

    def validation_data(self) -> pd.DataFrame:
        return self.holdout_lik.test_data()

    def data(self) -> pd.DataFrame:
        return self.training_data()

    # TODO: Review that this is correctly calculated
    def _conditional_log_likelihood(
        self, model: pbn.BayesianNetworkBase, df: pd.DataFrame | object
    ) -> float:
        df_pd = self._to_pandas(df)

        if pd.Series(df_pd[self.target]).isna().any():
            raise ValueError(
                "CLL cannot be computed with missing values in the target column."
            )

        eval_model = model.clone()
        eval_model.fit(self.training_data())

        y_true = pd.Series(df_pd[self.target]).to_numpy()
        log_joint = []
        for value in self._target_values:
            conditioned = (
                df_pd.copy()
            )  # TODO: Shouldn't this be conditioned with filter?
            conditioned[self.target] = value
            log_joint.append(np.asarray(eval_model.logl(conditioned), dtype=float))

        log_joint_matrix = np.vstack(log_joint)
        max_per_row = np.max(log_joint_matrix, axis=0)
        log_denom = max_per_row + np.log(
            np.exp(log_joint_matrix - max_per_row).sum(axis=0)
        )

        target_to_index = {value: idx for idx, value in enumerate(self._target_values)}
        true_idx = np.array([target_to_index[val] for val in y_true], dtype=int)
        num = log_joint_matrix[true_idx, np.arange(log_joint_matrix.shape[1])]

        return float(np.sum(num - log_denom))

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
# TODO: Metric-based structure learning
if __name__ == "__main__":
    hc = pbn.GreedyHillClimbing()
    start_model = pbn.GaussianNetwork(["a", "b", "c", "d"])
    learned_model = hc.estimate(
        operators=pbn.ArcOperatorSet(), score=OracleValidatedScore(), start=start_model
    )
    assert set(learned_model.arcs()) == {("a", "c"), ("b", "c"), ("c", "d")}

    # Short usage example for ConditionalLogLikelihoodValidatedScore.
    toy_df = pd.DataFrame(
        {
            "x1": [0.0, 0.2, 0.8, 1.1, 1.5, 1.9, 2.2, 2.5],
            "x2": [1.0, 0.9, 0.7, 0.4, 0.2, 0.1, -0.1, -0.3],
            "y": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    # TODO: Adapt to my BNCs
    X = toy_df[["x1", "x2"]]
    y = toy_df["y"]
    cll_score = ConditionalLogLikelihoodValidatedScore(
        toy_df, target="y", test_ratio=0.25, k=2, seed=7
    )
    toy_start = pbn.GaussianNetwork(["x1", "x2", "y"])
    toy_model = hc.estimate(
        operators=pbn.ArcOperatorSet(), score=cll_score, start=toy_start
    )
    print("Toy model arcs:", sorted(toy_model.arcs()))
    print("Toy train CLL:", cll_score.score(toy_model))
    print("Toy validation CLL:", cll_score.vscore(toy_model))
