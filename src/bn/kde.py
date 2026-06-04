import numpy as np
import pandas as pd
import pybnesian as pbn
from scipy.special import logsumexp

from .base import BayesianNetwork
from .semiparametric import SemiParametricBayesianNetwork


class KDEBayesianNetwork(SemiParametricBayesianNetwork):
    # bn_type = pbn.KDENetworkType() # NOTE: To allow hybrid KDE
    search_operators = ["arcs"]

    def __str__(self) -> str:
        """Returns the string representation of the SemiParametric Bayesian Network

        Returns:
            str: The string representation
        """
        return "KDE " + BayesianNetwork.__str__(self)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pbn.BayesianNetwork:
        nodes = X.columns
        self.type_whitelist = [
            (node, pbn.CKDEType()) for node in nodes if node != self.true_label
        ]
        super().fit(X, y)
        return self

    def posterior(
        self,
        query_nodes: list[str],
        evidence: dict[str, float],
        point: pd.Series,
        n_samples: int = 10_000,
        seed: int = 0,
    ) -> pd.Series:
        """
        Approximate p(query_nodes | evidence) using likelihood weighting.

        Parameters
        ----------
        bn : KDEBayesianNetwork
        evidence : dict[str, float]
            Observed variables, e.g. {"A": 1.2, "D": -0.4}
        query_nodes : list[str]
            Variables to return posterior samples for.
        n_samples : int
        seed : int

        Returns
        -------
        pd.Series
            Estimated posterior density for each query node evaluated at ``point``.
        """
        # 1. Initialize
        topo_order = list(self.graph().topological_sort())
        assignments = pd.DataFrame(index=np.arange(n_samples), columns=topo_order)
        log_weights = np.zeros(n_samples, dtype=float)
        # 2. Sample variables in topological order:
        for node_index, node in enumerate(topo_order):
            cpd = self.cpd(node)
            parents = cpd.evidence()
            parent_values = (
                assignments[parents]
                if len(parents) > 0
                else pd.DataFrame(index=assignments.index)
            )
            # 2a. If the variable is observed (part of the evidence), set its value to the observed value and update the weight by multiplying it with the conditional probability of the observed value given its parents.
            if node in evidence:
                observed_value = float(evidence[node])
                assignments[node] = observed_value

                point_df = parent_values.copy()
                point_df.insert(0, node, observed_value)
                log_weights += np.asarray(cpd.logl(point_df), dtype=float)
            # 2b. If the variable is not observed, sample its value from its conditional distribution given its parents.
            else:
                sampled_values = cpd.sample(
                    n_samples,
                    parent_values,
                    seed=seed + node_index,
                ).to_pandas()
                assignments[node] = sampled_values.to_numpy().reshape(-1)
        # 3. Store the Sample and its weight
        query_samples = assignments[query_nodes]
        weights = np.exp(log_weights - logsumexp(log_weights))

        posterior_values = {}
        # 5. Estimate Probabilities: Use the weighted samples to estimate probabilities or expectations.
        # Builds a weighted KDE per query variable
        for node in query_nodes:
            # For each node in query_nodes, it extracts the likelihood-weighted samples collected during sampling.
            samples = query_samples[node].to_numpy(dtype=float)
            # Chooses a bandwidth using Silverman’s rule of thumb: 1.06 * std * n**(-1/5). If that bandwidth is non-finite or ≤ 0, it falls back to max(std, 1.0).
            bandwidth = 1.06 * np.std(samples) * (len(samples) ** (-1.0 / 5.0))
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                bandwidth = max(np.std(samples), 1.0)
            # Computes Gaussian kernel values at the target point[node]
            normalized_deltas = (float(point[node]) - samples) / bandwidth
            kernel_values = np.exp(-0.5 * normalized_deltas**2) / (
                np.sqrt(2.0 * np.pi) * bandwidth
            )
            # Returns the weighted sum of kernels, i.e. sum(weights * kernel_values), which is the estimated posterior density p(node = point[node] | evidence).
            posterior_values[node] = float(np.sum(weights * kernel_values))
        # Returns a Pandas Series of densities (not normalized probabilities) for the requested query nodes.
        return pd.Series(posterior_values, index=query_nodes, dtype=float)

    # TODO: Calculate from posterior
    # def infer(
    #     self,
    #     evidence: dict[str, float] = {},
    #     json_file_path: Path | None = None,
    #     pdf_file_path: Path | None = None,
    # ) -> dict[str, dict]:
    #     """
    #     Performs inference on the Bayesian network using the provided evidence and target nodes.
    #     Args:
    #         evidence (dict[str, float], optional): A dictionary mapping node names to their observed values. Defaults to an empty dictionary. We can have hard evidence (e.g., {"Execution": True}) or soft evidence (e.g., {"Execution": [0.3, 0.9]}).
    #         json_file_path (Path | None, optional): If provided, exports the inference results to this file in JSON format.
    #         pdf_file_path (Path | None, optional): If provided, exports the graphical representation of the inference to this file in PDF format.
    #     Returns:
    #         dict[str, dict]: A dictionary where keys are node names and values are dictionaries containing the posterior probabilities for each state of the node.
    #     """
    #     result_dict = {}
    #     # ie = gclg.CLGVariableElimination(self.graphic)
    #     # ie.updateEvidence(evidence)

    #     # result_dict = {}
    #     # result_dict["structure"] = list(self.graphic.arcs())
    #     # result_dict["parameters"] = {}
    #     # for var_id, variable_name in enumerate(self.graphic.names()):
    #     #     post = ie.posterior(variable_name)
    #     #     result_dict["parameters"][var_id] = {
    #     #         "variable_name": variable_name,
    #     #         "probabilities": {
    #     #             "name": variable_name,
    #     #             "mean": post.mu(),
    #     #             "std": post.sigma(),
    #     #         },
    #     #     }

    #     # # export results
    #     # if json_file_path:
    #     #     with open(json_file_path, "w") as f:
    #     #         json.dump(result_dict, f, indent=4)
    #     # if pdf_file_path:
    #     #     gclgnb.exportInference(
    #     #         clg=self.graphic,
    #     #         filename=str(pdf_file_path),
    #     #         evs=evidence,
    #     #     )

    #     return result_dict
