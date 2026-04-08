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


if __name__ == "__main__":
    hc = pbn.GreedyHillClimbing()
    start_model = pbn.GaussianNetwork(["a", "b", "c", "d"])
    learned_model = hc.estimate(
        operators=pbn.ArcOperatorSet(), score=OracleValidatedScore(), start=start_model
    )
    assert set(learned_model.arcs()) == {("a", "c"), ("b", "c"), ("c", "d")}
