import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyagrum.clg as gclg
import pybnesian as pbn
from scipy.special import logsumexp
from scipy.stats import norm

from ..utils.constants import TRUE_ANOMALY_LABEL
from .base import BayesianNetwork


class SemiParametricBayesianNetwork(
    BayesianNetwork, pbn.SemiparametricBN
):  # Method Resolution Order important (save/load)
    """SemiParametric Bayesian Network class"""

    bn_type = pbn.SemiparametricBNType()
    search_operators = ["arcs", "node_type"]

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
        true_label: str = TRUE_ANOMALY_LABEL,
        prediction_label: str = "binary_predicted_label",
    ) -> None:
        """Initializes the SemiParametric Bayesian Network with the nodes, arcs, node_types and the structure learning parameters

        Args:
            nodes (list[str], optional): list of nodes. Defaults to [].
            arcs (list[tuple[str, str]], optional): list of arcs. Defaults to [].
            node_types (list[tuple[str, pbn.FactorType]], optional): list of node types. Defaults to [].
            search_score (str): Search score to be used for the structure learning. The possible scores ((validate_options.cpp)) are:
                - "cv-lik" (Cross-Validated likelihood)
                - "holdout-lik" (Hold-out likelihood)
                - "validated-lik" (Validated likelihood with cross-validation). Defaults to "validated-lik".
            arc_blacklist (list[tuple[str, str]], optional): Arc blacklist (forbidden arcs). Defaults to [].
            arc_whitelist (list[tuple[str, str]], optional): Arc whitelist (forced arcs). Defaults to [].
            type_blacklist (list[tuple[str, pbn.FactorType]], optional): Node type blacklist (forbidden node types). Defaults to [].
            type_whitelist (list[tuple[str, pbn.FactorType]], optional): Node type whitelist (forced node types). Defaults to [].
            max_indegree (int, optional): Maximum indegree allowed in the graph. Defaults to 0.
            max_iters (int, optional): Maximum number of search iterations. Defaults to 2147483647.
            epsilon (int, optional): Minimum delta score allowed for each operator. If the new operator is less than epsilon, the search process is stopped. Defaults to 0.
            patience (int, optional): The patience parameter (only used with pbn.ValidatedScore). Defaults to 0.
            seed (int | None, optional): Seed parameter of the score (if needed). Defaults to None.
            num_folds (int, optional): Number of folds for the CVLikelihood and ValidatedLikelihood scores. Defaults to 5.
            test_holdout_ratio (float, optional): Parameter for the HoldoutLikelihood and ValidatedLikelihood scores. Defaults to 0.2.
            max_train_data_size (int, optional): Maximum sample size to be used for the structure learning. Defaults to 0.
            verbose (bool, optional): If True the progress will be displayed, otherwise nothing will be displayed. Defaults to False.
            true_label (str, optional): The true label column name. Defaults to TRUE_ANOMALY_LABEL.
            prediction_label (str, optional): The predicted label column name. Defaults to "binary_predicted_label".
        """
        pbn.SemiparametricBN.__init__(self, nodes=[])
        BayesianNetwork.__init__(
            self,
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
        )

    def __str__(self) -> str:
        """Returns the string representation of the SemiParametric Bayesian Network

        Returns:
            str: The string representation
        """
        return "SemiParametric " + BayesianNetwork.__str__(self)

    def _fit_parameters(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pbn.BayesianNetwork:
        data = pd.concat([X, y], axis=1)
        pbn.SemiparametricBN.fit(self, data)
        return self

    # def infer(
    #     self,
    #     evidence: dict[str, float] = {},
    #     json_file_path: Path | None = None,
    #     pdf_file_path: Path | None = None,
    #     n_samples: int = 1000,
    #     seed: int = 0,
    # ) -> dict[str, dict]:
    #     """
    #     Performs likelihood weighting inference on the Bayesian network using the provided evidence and target nodes.
    #     Args:
    #         evidence (dict[str, float], optional): A dictionary mapping node names to their observed values. Defaults to an empty dictionary. We can have hard evidence (e.g., {"Execution": True}) or soft evidence (e.g., {"Execution": [0.3, 0.9]}).
    #         n_samples (int, optional): The number of samples to draw for the likelihood weighting inference. Defaults to 10,000.
    #         seed (int, optional): The random seed for reproducibility. Defaults to 0.
    #         json_file_path (Path | None, optional): If provided, exports the inference results to this file in JSON format.
    #         pdf_file_path (Path | None, optional): If provided, exports the graphical representation of the inference to this file in PDF format.
    #     Returns:
    #         dict[str, dict]: A dictionary containing the structure of the Bayesian network and the parameters of the inference results. The structure is represented as a list of arcs, and the parameters include the weights and assignments from the likelihood weighting inference.
    #     """
    #     # Initialize batched assignments and per-sample log weights.
    #     topo_order = list(self.graph().topological_sort())
    #     assignments = pd.DataFrame(index=np.arange(n_samples), columns=topo_order)
    #     log_weights = np.zeros(n_samples, dtype=float)

    #     # Sample variables in topological order and accumulate log weights.
    #     for node_index, node in enumerate(topo_order):
    #         cpd = self.cpd(node)
    #         parents = cpd.evidence()
    #         parent_values = (
    #             assignments[parents]
    #             if len(parents) > 0
    #             else pd.DataFrame(index=assignments.index)
    #         )
    #         # Clamp evidence and add its log-likelihood under the current parents.
    #         if node in evidence:
    #             observed_value = float(evidence[node])
    #             assignments[node] = observed_value

    #             point_df = parent_values.copy()
    #             point_df.insert(0, node, observed_value)
    #             log_weights += np.asarray(cpd.logl(point_df), dtype=float)
    #         # Sample all rows at once from the conditional distribution of this node.
    #         else:
    #             assignments[node] = cpd.sample(
    #                 n_samples,
    #                 parent_values,
    #                 seed=seed + node_index,
    #             ).to_pandas()
    #     # Normalize log weights to get importance weights.
    #     weights = np.exp(log_weights - logsumexp(log_weights))
    #     infer_dict = {
    #         "structure": self.arcs(),
    #         "parameters": {
    #             "weights": weights,
    #             "assignments": assignments,
    #         },
    #     }
    #     # export results
    #     if json_file_path:
    #         export_dict = infer_dict.copy()
    #         export_dict["parameters"]["weights"] = weights.tolist()
    #         export_dict["parameters"]["assignments"] = assignments.to_dict(
    #             orient="list"
    #         )
    #         with open(json_file_path, "w") as f:
    #             json.dump(export_dict, f, indent=4)
    #     if pdf_file_path:
    #         self.save(pdf_file_path)
    #     return infer_dict

    # RFE: Allow automatic sampling stop criterion
    def infer(
        self,
        evidence: dict[str, float] = {},
        json_file_path: Path | None = None,
        pdf_file_path: Path | None = None,
        n_samples: int = 1000,
        seed: int = 0,
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
        infer_dict = {"structure": self.arcs(), "parameters": {}}

        # Sample variables in topological order and accumulate log weights.
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
                if cpd.type() == pbn.CKDEType():
                    point_df = parent_values.copy()
                    point_df.insert(0, node, observed_value)
                    log_weights += np.asarray(cpd.logl(point_df), dtype=float)
                elif cpd.type() == pbn.LinearGaussianCPDType():
                    pass
            # Sample all rows at once from the conditional distribution of this node.
            else:
                if cpd.type() == pbn.CKDEType():
                    assignments[node] = cpd.sample(
                        n_samples,
                        parent_values,
                        seed=seed + node_index,
                    ).to_pandas()

        # RFE: Remove true_label for BNCs
        # If the true label is in the graph, we need to remove it from the evidence and create an auxiliary graph without it for inference
        # evidence = {k: v for k, v in evidence.items() if k != self.true_label}

        # The problem is that the arc should have a coefficient?
        clg_subgraphic = gclg.CLG()
        # Copies the nodes to the pyagrum graphic, for non-CLG nodes, we set the mean and std to 0 and 1 respectively, since they are not used in the inference
        for node in self.nodes():
            # if node != self.true_label:
            cpd = self.cpd(node)
            if self.cpd(node).type() == pbn.LinearGaussianCPDType():
                mu = cpd.beta[0]
                std = np.sqrt(cpd.variance)
            else:
                mu = 0.0
                std = 1.0
            clg_subgraphic.add(gclg.GaussianVariable(node, mu, std))
        # Copies the arcs to the pyagrum graphic for CLG nodes, and sets the coefficients for the arcs
        for source, target in self.arcs():
            cpd = self.cpd(target)
            if self.cpd(target).type() == pbn.LinearGaussianCPDType():
                parents = cpd.evidence()
                parent_index = parents.index(source)
                coef = cpd.beta[parent_index + 1]
                clg_subgraphic.addArc(source, target, coef)
        ie = gclg.CLGVariableElimination(clg_subgraphic)
        unique_evidence = assignments.drop_duplicates().dropna(axis=1)

        # NOTE: Expensive inference for CLG nodes, we need to compute the posterior for each unique evidence and assign it to the corresponding rows in the assignments dataframe
        for _, row in unique_evidence.iterrows():
            aux_evidence = row.to_dict()
            # We put the CKDE evidence in the CLG inference
            ie.updateEvidence(aux_evidence)  # This overwrites all evidence
            for node in self.feature_names_in_:
                if (
                    node not in evidence
                    and self.cpd(node).type() == pbn.LinearGaussianCPDType()
                ):
                    post = ie.posterior(node)
                    # Assign to node where unique_evidence is the same
                    assignments.loc[
                        assignments[list(aux_evidence.keys())].eq(row).all(axis=1), node
                    ] = post
        # Normalize log weights to get importance weights.
        weights = np.exp(log_weights - logsumexp(log_weights))
        infer_dict["parameters"] = {
            "weights": weights,
            "assignments": assignments,
        }
        # export results
        if json_file_path:
            export_dict = infer_dict.copy()
            export_dict["parameters"]["weights"] = weights.tolist()
            export_dict["parameters"]["assignments"] = assignments.to_dict(
                orient="list"
            )
            with open(json_file_path, "w") as f:
                json.dump(export_dict, f, indent=4)
        if pdf_file_path:
            self.save(pdf_file_path)
        return infer_dict

    def posterior(
        self,
        query_node: str,
        evidence: dict[str, float],
        point: pd.Series,
        n_samples: int = 1000,
        seed: int = 0,
        likelihood_weighting_dict: dict[str, dict] = {},
    ) -> float:
        """
        Approximate the posterior density of query nodes at ``point`` using likelihood weighting.
        Uses a weighted Gaussian kernel density estimation (KDE) to estimate the posterior density of the query nodes given the evidence and the point at which to evaluate the density.
        The likelihood weighting inference is performed to obtain samples and weights, which are then used to compute the KDE.
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
        likelihood_weighting_dict : dict[str, dict]
            Optional dictionary containing the results of a previous likelihood weighting inference. If provided, it will be used to avoid redundant computations. Defaults to an empty dictionary.
        Returns
        -------
        float
            The estimated posterior density of the query node at the specified point.
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

        cpd = self.cpd(query_node)
        if cpd.type() == pbn.LinearGaussianCPDType():
            # Assign for CLG nodes depending on the matching evidence
            samples_post = assignments[query_node]
            # For each of the rows, we need to compute the posterior mean and std based on the evidence
            mu = samples_post.apply(lambda x: x.mu())
            std = samples_post.apply(lambda x: x.sigma())
            kernel_values = norm.pdf(point[query_node], loc=mu, scale=std)
        else:
            # Estimate the density at ``point`` with a weighted Gaussian KDE per query node.
            samples = assignments[query_node].to_numpy(dtype=float)
            kernel_values = _kernel_values(samples)

        # Weighted sum of kernels
        posterior_value = np.sum(weights * kernel_values)
        return posterior_value
