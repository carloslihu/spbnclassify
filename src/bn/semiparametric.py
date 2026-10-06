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

    def _parent_values(self, node: str, assignments: pd.DataFrame) -> pd.DataFrame:
        """Returns the parent columns of ``node`` (an empty frame with the same index for root nodes)"""
        parents = self.cpd(node).evidence()
        return (
            assignments[parents]
            if len(parents) > 0
            else pd.DataFrame(index=assignments.index)
        )

    def _node_logl(self, node: str, assignments: pd.DataFrame) -> np.ndarray:
        """Row-wise log-likelihood of ``node`` given its parents under its conditional distribution"""
        point_df = self._parent_values(node, assignments).copy()
        point_df.insert(0, node, assignments[node].to_numpy(dtype=float))
        return np.asarray(self.cpd(node).logl(point_df), dtype=float)

    def _rao_blackwellize_clg(
        self, assignments: pd.DataFrame, evidence: dict[str, float]
    ) -> pd.DataFrame:
        """Replaces the non-evidence LinearGaussian columns with their exact CLG posterior
        (pyagrum) conditioned on the CKDE and evidence values of each row.

        Args:
            assignments (pd.DataFrame): Sampled assignments (one row per sample).
            evidence (dict[str, float]): Observed variables.

        Returns:
            pd.DataFrame: The assignments with CLG posterior objects in the non-evidence LinearGaussian columns.
        """
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

        # Condition on CKDE and evidence columns only (non-evidence CLG nodes are the query)
        clg_query_nodes = [
            node
            for node in self.feature_names_in_
            if node not in evidence
            and self.cpd(node).type() == pbn.LinearGaussianCPDType()
        ]
        conditioning_columns = [
            node for node in assignments.columns if node not in clg_query_nodes
        ]
        unique_evidence = (
            assignments[conditioning_columns].drop_duplicates().dropna(axis=1)
        )
        # The CLG posteriors are objects, so the target columns must allow them
        for node in clg_query_nodes:
            assignments[node] = assignments[node].astype(object)

        # NOTE: Expensive inference for CLG nodes, we need to compute the posterior for each unique evidence and assign it to the corresponding rows in the assignments dataframe
        for _, row in unique_evidence.iterrows():
            aux_evidence = row.to_dict()
            # We put the CKDE evidence in the CLG inference
            ie.updateEvidence(aux_evidence)  # This overwrites all evidence
            row_mask = assignments[list(aux_evidence.keys())].eq(row).all(axis=1)
            for node in clg_query_nodes:
                post = ie.posterior(node)
                # Assign to node where unique_evidence is the same
                assignments.loc[row_mask, node] = post
        return assignments

    def _export_inference(
        self,
        infer_dict: dict[str, dict],
        json_file_path: Path | None = None,
        pdf_file_path: Path | None = None,
    ) -> None:
        """Exports the inference results to JSON and/or the network graph to PDF"""
        if json_file_path:
            export_dict = infer_dict.copy()
            export_dict["parameters"]["weights"] = infer_dict["parameters"][
                "weights"
            ].tolist()
            export_dict["parameters"]["assignments"] = infer_dict["parameters"][
                "assignments"
            ].to_dict(orient="list")
            with open(json_file_path, "w") as f:
                json.dump(export_dict, f, indent=4)
        if pdf_file_path:
            self.save(pdf_file_path)

    def _posterior_density(
        self,
        query_node: str,
        point: pd.Series,
        assignments: pd.DataFrame,
        weights: np.ndarray,
    ) -> float:
        """Estimates the posterior density of ``query_node`` at ``point`` from weighted samples.
        CKDE nodes use a weighted Gaussian KDE, CLG nodes a weighted mixture of their CLG posteriors.
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
            parent_values = self._parent_values(node, assignments)
            # Clamp evidence and add its log-likelihood under the current parents.
            if node in evidence:
                observed_value = float(evidence[node])
                assignments[node] = observed_value
                if cpd.type() == pbn.CKDEType():
                    log_weights += self._node_logl(node, assignments)
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

        assignments = self._rao_blackwellize_clg(assignments, evidence)

        # Normalize log weights to get importance weights.
        weights = np.exp(log_weights - logsumexp(log_weights))
        infer_dict["parameters"] = {
            "weights": weights,
            "assignments": assignments,
        }
        self._export_inference(infer_dict, json_file_path, pdf_file_path)
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
        if likelihood_weighting_dict == {}:
            # Use provided likelihood weighting results if available
            likelihood_weighting_dict = self.infer(
                evidence=evidence, n_samples=n_samples, seed=seed
            )

        return self._posterior_density(
            query_node,
            point,
            likelihood_weighting_dict["parameters"]["assignments"],
            likelihood_weighting_dict["parameters"]["weights"],
        )

    # RFE: Allow automatic convergence diagnostics (e.g., R-hat) as stop criterion
    def infer_GS(
        self,
        evidence: dict[str, float] = {},
        json_file_path: Path | None = None,
        pdf_file_path: Path | None = None,
        n_samples: int = 1000,
        n_chains: int = 100,
        burn_in: int = 100,
        thinning: int = 1,
        seed: int = 0,
    ) -> dict[str, dict]:
        """
        Performs Gibbs sampling inference (Metropolis-within-Gibbs) on the Bayesian network using the provided evidence.
        CKDE full conditionals have no closed form, so each unobserved node X is updated with a Metropolis-Hastings step whose proposal is its own CPD p(X | Pa(X)).
        The proposal cancels the p(X | Pa(X)) term, so the acceptance ratio only involves the children of X:
            alpha = min(1, prod_c p(c | Pa(c), X=x*) / prod_c p(c | Pa(c), X=x))
        ``n_chains`` independent chains are run in batch. LinearGaussian nodes are finally Rao-Blackwellized with their exact CLG posterior, as in ``infer``.
        Args:
            evidence (dict[str, float], optional): A dictionary mapping node names to their observed values. Defaults to an empty dictionary.
            json_file_path (Path | None, optional): If provided, exports the inference results to this file in JSON format.
            pdf_file_path (Path | None, optional): If provided, exports the graphical representation of the inference to this file in PDF format.
            n_samples (int, optional): The total number of samples (over all chains) to keep after burn-in and thinning. Defaults to 1000.
            n_chains (int, optional): The number of independent chains run in parallel. Defaults to 100.
            burn_in (int, optional): The number of initial sweeps discarded in each chain. Defaults to 100.
            thinning (int, optional): Keep one sweep every ``thinning`` sweeps after burn-in. Defaults to 1.
            seed (int, optional): The random seed for reproducibility. Defaults to 0.
        Returns:
            dict[str, dict]: A dictionary containing the structure of the Bayesian network and the parameters of the inference results. The parameters include the (uniform) weights, the assignments and the per-node acceptance rate of the Gibbs sampler.
        """
        rng = np.random.default_rng(seed)

        def _next_seed() -> int:
            return int(rng.integers(2**31 - 1))

        topo_order = list(self.graph().topological_sort())
        free_nodes = [node for node in topo_order if node not in evidence]
        children = {node: list(self.children(node)) for node in free_nodes}
        infer_dict = {"structure": self.arcs(), "parameters": {}}

        # Initialize the chains by forward sampling with the evidence clamped.
        state = pd.DataFrame(index=np.arange(n_chains), columns=topo_order, dtype=float)
        for node in topo_order:
            if node in evidence:
                state[node] = float(evidence[node])
            else:
                state[node] = (
                    self.cpd(node)
                    .sample(
                        n_chains, self._parent_values(node, state), seed=_next_seed()
                    )
                    .to_pandas()
                    .to_numpy(dtype=float)
                )

        n_kept_sweeps = int(np.ceil(n_samples / n_chains))
        n_sweeps = burn_in + n_kept_sweeps * thinning
        accepted = {node: 0 for node in free_nodes}
        kept_states = []

        for sweep in range(n_sweeps):
            for node in free_nodes:
                cpd = self.cpd(node)
                proposal = (
                    cpd.sample(
                        n_chains, self._parent_values(node, state), seed=_next_seed()
                    )
                    .to_pandas()
                    .to_numpy(dtype=float)
                )
                # Acceptance ratio only depends on the children (Markov blanket) of the node.
                log_alpha = np.zeros(n_chains, dtype=float)
                if len(children[node]) > 0:
                    proposed_state = state.copy()
                    proposed_state[node] = proposal
                    for child in children[node]:
                        log_alpha += self._node_logl(child, proposed_state)
                        log_alpha -= self._node_logl(child, state)
                log_alpha = np.nan_to_num(log_alpha, nan=-np.inf)
                accept = np.log(rng.random(n_chains)) < log_alpha
                state.loc[accept, node] = proposal[accept]
                accepted[node] += int(accept.sum())

            if sweep >= burn_in and (sweep - burn_in) % thinning == thinning - 1:
                kept_states.append(state.copy())

        assignments = pd.concat(kept_states, ignore_index=True).iloc[:n_samples]
        assignments = assignments.reset_index(drop=True)
        assignments = self._rao_blackwellize_clg(assignments, evidence)

        # MCMC samples are equally weighted.
        weights = np.full(len(assignments), 1.0 / len(assignments))
        infer_dict["parameters"] = {
            "weights": weights,
            "assignments": assignments,
            "acceptance_rate": {
                node: accepted[node] / (n_sweeps * n_chains) for node in free_nodes
            },
        }
        self._export_inference(infer_dict, json_file_path, pdf_file_path)
        return infer_dict

    def posterior_GS(
        self,
        query_node: str,
        evidence: dict[str, float],
        point: pd.Series,
        n_samples: int = 1000,
        n_chains: int = 100,
        burn_in: int = 100,
        thinning: int = 1,
        seed: int = 0,
        gibbs_sampling_dict: dict[str, dict] = {},
    ) -> float:
        """
        Approximate the posterior density of query nodes at ``point`` using Gibbs sampling.
        Uses a Gaussian kernel density estimation (KDE) over the Gibbs samples for CKDE nodes, and a mixture of the exact CLG posteriors for LinearGaussian nodes.
        Parameters
        ----------
        query_node : str
            Variable to return posterior samples for.
        evidence : dict[str, float]
            Observed variables, e.g. {"A": 1.2, "D": -0.4}
        point : pd.Series
            The point at which to evaluate the posterior density, e.g. pd.Series({"A": 1.0, "D": -0.5})
        n_samples : int
            The total number of Gibbs samples to keep. Defaults to 1000.
        n_chains : int
            The number of independent chains run in parallel. Defaults to 100.
        burn_in : int
            The number of initial sweeps discarded in each chain. Defaults to 100.
        thinning : int
            Keep one sweep every ``thinning`` sweeps after burn-in. Defaults to 1.
        seed : int
            The random seed for reproducibility. Defaults to 0.
        gibbs_sampling_dict : dict[str, dict]
            Optional dictionary containing the results of a previous Gibbs sampling inference. If provided, it will be used to avoid redundant computations. Defaults to an empty dictionary.
        Returns
        -------
        float
            The estimated posterior density of the query node at the specified point.
        """
        if gibbs_sampling_dict == {}:
            gibbs_sampling_dict = self.infer_GS(
                evidence=evidence,
                n_samples=n_samples,
                n_chains=n_chains,
                burn_in=burn_in,
                thinning=thinning,
                seed=seed,
            )

        return self._posterior_density(
            query_node,
            point,
            gibbs_sampling_dict["parameters"]["assignments"],
            gibbs_sampling_dict["parameters"]["weights"],
        )
