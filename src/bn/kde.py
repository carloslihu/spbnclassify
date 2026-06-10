from pathlib import Path

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

    # RFE: Unify format with other infer methods
    def infer(
        self,
        evidence: dict[str, float] = {},
        n_samples: int = 10_000,
        seed: int = 0,
        json_file_path: Path | None = None,
        pdf_file_path: Path | None = None,
    ) -> dict[str, dict]:
        """
        Performs likelihood weighting inference on the Bayesian network using the provided evidence and target nodes.
        Args:
            evidence (dict[str, float], optional): A dictionary mapping node names to their observed values. Defaults to an empty dictionary. We can have hard evidence (e.g., {"Execution": True}) or soft evidence (e.g., {"Execution": [0.3, 0.9]}).
            n_samples (int, optional): The number of samples to draw for the likelihood weighting inference. Defaults to 10,000.
            seed (int, optional): The random seed for reproducibility. Defaults to 0.
            json_file_path (Path | None, optional): If provided, exports the inference results to this file in JSON format.
            pdf_file_path (Path | None, optional): If provided, exports the graphical representation of the inference to this file in PDF format.
        Returns:
            dict[str, dict]: A dictionary containing the structure of the Bayesian network and the parameters of the inference results. The structure is represented as a list of arcs, and the parameters include the weights and assignments from the likelihood weighting inference.
        """
        # Initialize batched assignments and per-sample log weights.
        topo_order = list(self.graph().topological_sort())
        assignments = pd.DataFrame(index=np.arange(n_samples), columns=topo_order)
        log_weights = np.zeros(n_samples, dtype=float)

        # Sample variables in topological order and accumulate likelihood weights.
        for node_index, node in enumerate(topo_order):
            cpd = self.cpd(node)
            parents = cpd.evidence()
            parent_values = (
                assignments[parents]
                if len(parents) > 0
                else pd.DataFrame(index=assignments.index)
            )
            # Clamp evidence and add its log-likelihood under the current parents.
            if node in evidence:
                observed_value = float(evidence[node])
                assignments[node] = observed_value

                point_df = parent_values.copy()
                point_df.insert(0, node, observed_value)
                log_weights += np.asarray(cpd.logl(point_df), dtype=float)
            # Sample all rows at once from the conditional distribution of this node.
            else:
                sampled_values = cpd.sample(
                    n_samples,
                    parent_values,
                    seed=seed + node_index,
                ).to_pandas()
                assignments[node] = sampled_values.to_numpy().reshape(-1)
        # Normalize log weights to get importance weights.
        weights = np.exp(log_weights - logsumexp(log_weights))
        result_dict = {
            "structure": list(self.graphic.arcs()),
            "parameters": {
                "weights": weights,
                "assignments": assignments,
            },
        }
        for var_id, variable_name in enumerate(self.graphic.names()):
            result_dict["parameters"][var_id] = {
                "variable_name": variable_name,
            }
        # TODO: Export results
        # # export results
        # if json_file_path:
        #     with open(json_file_path, "w") as f:
        #         json.dump(result_dict, f, indent=4)
        # if pdf_file_path:
        #     gclgnb.exportInference(
        #         clg=self.graphic,
        #         filename=str(pdf_file_path),
        #         evs=evidence,
        #     )
        return result_dict

    def posterior(
        self,
        query_node: str,
        evidence: dict[str, float],
        point: pd.Series,
        n_samples: int = 10_000,
        seed: int = 0,
        likelihood_weighting_dict: dict[str, dict] = {},
    ) -> pd.Series:
        """
        Approximate the posterior density of query nodes at ``point`` using likelihood weighting.

        Parameters
        ----------
        query_node : str
            Variable to return posterior samples for.
        evidence : dict[str, float]
            Observed variables, e.g. {"A": 1.2, "D": -0.4}
        point : pd.Series
            The point at which to evaluate the posterior density, e.g. pd.Series({"A": 1.0, "D": -0.5})
        n_samples : int
            The number of samples to draw for the likelihood weighting inference. Defaults to 10,000.
        seed : int
            The random seed for reproducibility. Defaults to 0.

        Returns
        -------
        pd.Series
            Estimated posterior density for each query node evaluated at ``point``.
        """

        def _kernel_values(samples: np.ndarray) -> np.ndarray:
            # Silverman’s rule of thumb: 1.06 * std * n**(-1/5)
            bandwidth = 1.06 * np.std(samples) * (len(samples) ** (-1.0 / 5.0))
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                bandwidth = max(np.std(samples), 1.0)

            # Computes Gaussian kernel values at the target point[query_node]
            normalized_deltas = (float(point[query_node]) - samples) / bandwidth
            kernel_values = np.exp(-0.5 * normalized_deltas**2) / (
                np.sqrt(2.0 * np.pi) * bandwidth
            )
            return kernel_values

        if likelihood_weighting_dict == {}:
            # Use provided likelihood weighting results if available
            likelihood_weighting_dict = self.infer(
                evidence=evidence, n_samples=n_samples, seed=seed
            )
        assignments = likelihood_weighting_dict["parameters"]["assignments"]
        weights = likelihood_weighting_dict["parameters"]["weights"]

        # Estimate the density at ``point`` with a weighted Gaussian KDE per query node.
        samples = assignments[query_node].to_numpy(dtype=float)
        kernel_values = _kernel_values(samples)

        # Weighted sum of kernels
        posterior_values = np.sum(weights * kernel_values)

        return posterior_values
