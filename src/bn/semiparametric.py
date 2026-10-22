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

    def _sampled_nodes(self) -> set[str]:
        """Returns the CKDE nodes and their (LinearGaussian) ancestors, i.e. the ancestor-closed set
        that must be sampled to keep the topological order in likelihood weighting"""
        sampled = {
            node for node in self.nodes() if self.cpd(node).type() == pbn.CKDEType()
        }
        for node in reversed(list(self.graph().topological_sort())):
            if node in sampled:
                sampled.update(self.parents(node))
        return sampled

    def _clg_inference_engine(self) -> gclg.CLGVariableElimination:
        """Builds the pyagrum CLG inference engine of the LinearGaussian part of the network"""
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
        return gclg.CLGVariableElimination(clg_subgraphic)

    def _clg_evidence_logl(
        self,
        assignments: pd.DataFrame,
        evidence: dict[str, float],
        sampled_nodes: set[str],
    ) -> np.ndarray:
        """Row-wise log p(e_R | S) of the evidence on the non-sampled LinearGaussian nodes R given the sampled values S,
        marginalizing the non-evidence nodes of R. Since S is ancestor-closed, R given S is a linear Gaussian network,
        so it is computed exactly with the chain rule sum_j log p(e_j | S, e_1..e_{j-1}).
        """
        log_lik = np.zeros(len(assignments), dtype=float)
        rb_evidence = [
            node
            for node in self.graph().topological_sort()
            if node in evidence and node not in sampled_nodes
        ]
        if len(rb_evidence) == 0:
            return log_lik

        ie = self._clg_inference_engine()

        def _chain_logl(conditioning: dict[str, float]) -> float:
            value = 0.0
            # updateEvidence merges into the previous evidence, so the previous row's evidence must be erased
            ie.eraseAllEvidence()
            for node in rb_evidence:
                ie.updateEvidence(conditioning)
                post = ie.posterior(node)
                value += norm.logpdf(
                    float(evidence[node]), loc=post.mu(), scale=post.sigma()
                )
                conditioning[node] = float(evidence[node])
            return value

        sampled_columns = [node for node in assignments.columns if node in sampled_nodes]
        if len(sampled_columns) == 0:
            log_lik[:] = _chain_logl({})
            return log_lik

        unique_samples = assignments[sampled_columns].drop_duplicates()
        for _, row in unique_samples.iterrows():
            row_mask = assignments[sampled_columns].eq(row).all(axis=1).to_numpy()
            log_lik[row_mask] = _chain_logl(row.to_dict())
        return log_lik

    def _rao_blackwellize_clg(
        self,
        assignments: pd.DataFrame,
        evidence: dict[str, float],
        sampled_nodes: set[str] = frozenset(),
    ) -> pd.DataFrame:
        """Replaces the non-evidence, non-sampled LinearGaussian columns with their exact CLG posterior
        (pyagrum) conditioned on the sampled and evidence values of each row.

        Args:
            assignments (pd.DataFrame): Sampled assignments (one row per sample).
            evidence (dict[str, float]): Observed variables.
            sampled_nodes (set[str], optional): LinearGaussian nodes whose sampled values are kept and conditioned on. Defaults to none.

        Returns:
            pd.DataFrame: The assignments with CLG posterior objects in the non-evidence, non-sampled LinearGaussian columns.
        """
        ie = self._clg_inference_engine()

        # Condition on sampled and evidence columns only (the remaining CLG nodes are the query)
        clg_query_nodes = [
            node
            for node in self.nodes()
            if node not in evidence
            and node not in sampled_nodes
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
            # We put the CKDE evidence in the CLG inference (updateEvidence merges, so the previous row's evidence is erased)
            ie.eraseAllEvidence()
            ie.updateEvidence(aux_evidence)
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
            # Builds new dicts so the returned infer_dict is not modified
            assignments = infer_dict["parameters"]["assignments"].copy()
            # Rao-Blackwellized CLG columns hold pyagrum posteriors, exported as their mean and std
            for node in assignments.columns:
                if not pd.api.types.is_numeric_dtype(assignments[node]):
                    assignments[node] = assignments[node].apply(
                        lambda post: {"mu": post.mu(), "sigma": post.sigma()}
                    )
            export_dict = {
                **infer_dict,
                "parameters": {
                    **infer_dict["parameters"],
                    "weights": infer_dict["parameters"]["weights"].tolist(),
                    "assignments": assignments.to_dict(orient="list"),
                },
            }
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
        Sampled (numeric) columns use a weighted Gaussian KDE, Rao-Blackwellized CLG columns a weighted mixture of their CLG posteriors.
        """

        def _kernel_values(samples: np.ndarray) -> np.ndarray:
            # Weighted Silverman’s rule of thumb: 1.06 * std_w * n_eff**(-1/5), with the effective sample size n_eff = 1 / sum(w**2)
            weighted_mean = np.sum(weights * samples)
            weighted_std = np.sqrt(np.sum(weights * (samples - weighted_mean) ** 2))
            n_eff = 1.0 / np.sum(weights**2)
            bandwidth = 1.06 * weighted_std * (n_eff ** (-1.0 / 5.0))
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                bandwidth = max(weighted_std, 1.0)

            # Computes Gaussian kernel values at the target point[query_node]
            normalized_deltas = (float(point[query_node]) - samples) / bandwidth
            kernel_values = np.exp(-0.5 * normalized_deltas**2) / (
                np.sqrt(2.0 * np.pi) * bandwidth
            )

            return kernel_values

        if not pd.api.types.is_numeric_dtype(assignments[query_node]):
            # Assign for Rao-Blackwellized CLG nodes depending on the matching evidence
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
        It is Rao-Blackwellized: only the CKDE nodes and their LinearGaussian ancestors are sampled, and the remaining LinearGaussian nodes are replaced by their exact CLG posterior.
        The weights include the evidence likelihood of the sampled nodes and the marginal likelihood of the evidence on the remaining LinearGaussian nodes.
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
        infer_dict = {
            "structure": self.arcs(),
            "evidence": dict(evidence),
            "parameters": {},
        }

        # CKDE nodes and their CLG ancestors are sampled, the rest of CLG nodes are Rao-Blackwellized.
        sampled = self._sampled_nodes()

        # Sample variables in topological order and accumulate log weights.
        for node_index, node in enumerate(topo_order):
            if node not in sampled:
                continue
            cpd = self.cpd(node)
            # Clamp evidence and add its log-likelihood under the current parents.
            if node in evidence:
                assignments[node] = float(evidence[node])
                log_weights += self._node_logl(node, assignments)
            # Sample all rows at once from the conditional distribution of this node.
            else:
                assignments[node] = cpd.sample(
                    n_samples,
                    self._parent_values(node, assignments),
                    seed=seed + node_index,
                ).to_pandas()

        # Evidence on the Rao-Blackwellized CLG nodes, with their non-evidence nodes marginalized.
        log_weights += self._clg_evidence_logl(assignments, evidence, sampled)
        for node in evidence:
            if node not in sampled:
                assignments[node] = float(evidence[node])
        assignments = self._rao_blackwellize_clg(assignments, evidence, sampled)

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

        Raises
        ------
        ValueError
            If the query node is in the evidence, or if ``likelihood_weighting_dict`` was computed with a different evidence.
        """
        if query_node in evidence:
            raise ValueError("Query node and evidence nodes must be disjoint.")
        if (
            likelihood_weighting_dict != {}
            and likelihood_weighting_dict.get("evidence") != evidence
        ):
            raise ValueError(
                "likelihood_weighting_dict was computed with a different evidence."
            )
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

    def _kernel_cache(self, node: str) -> dict:
        """Precomputes the Gaussian-mixture representation of the CPD of ``node``.

        A CKDE conditional p(x | pa) = f(x, pa) / f(pa) with Gaussian kernels of bandwidth H is the mixture
        sum_j w_j(pa) N(x; m_j(pa), s2), with w_j ∝ N(pa; pa_j, H_pp), m_j = x_j + H_xp H_pp^-1 (pa - pa_j)
        and s2 = H_xx - H_xp H_pp^-1 H_px. A LinearGaussian CPD is the single component N(x; beta0 + beta·pa, variance).
        """
        cpd = self.cpd(node)
        if cpd.type() == pbn.LinearGaussianCPDType():
            beta = np.asarray(cpd.beta, dtype=float)
            return {
                "type": "lg",
                "parents": cpd.evidence(),
                "beta0": beta[0],
                "betas": beta[1:],
                "var": float(cpd.variance),
            }

        kde = cpd.kde_joint()
        variables = kde.variables()  # [node, *parents]
        H = np.atleast_2d(np.asarray(kde.bandwidth, dtype=float))
        D = kde.dataset().to_pandas()[variables].to_numpy(dtype=float)
        parents = variables[1:]
        x_train = D[:, 0]
        if len(parents) == 0:
            s2 = H[0, 0]
            cache = {"whiten": None, "reg": np.zeros(0), "offset": x_train}
        else:
            H_pp_inv = np.linalg.inv(H[1:, 1:])
            reg = H[0, 1:] @ H_pp_inv
            s2 = H[0, 0] - reg @ H[1:, 0]
            # H_pp^-1 = A A^T, so (pa - pa_j)^T H_pp^-1 (pa - pa_j) = ||pa A - pa_j A||^2
            whiten = np.linalg.cholesky(H_pp_inv)
            z_train = D[:, 1:] @ whiten
            cache = {
                "whiten": whiten,
                "z_train": z_train,
                "z_train_sq": np.sum(z_train**2, axis=1),
                "reg": reg,
                "offset": x_train - D[:, 1:] @ reg,
            }
        cache.update(
            {
                "type": "ckde",
                "parents": parents,
                "var": float(s2),
                # Each conditional is a mixture of Gaussians with common variance s2, hence bounded
                "log_bound": -0.5 * np.log(2.0 * np.pi * s2),
            }
        )
        return cache

    @staticmethod
    def _conditional_mixture(
        kernel: dict, state: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns the (unnormalized) log-weights and means, both (n_chains, n_components), and the common
        variance of the Gaussian mixture p(x | pa) for every chain in ``state``."""
        pa = state[kernel["parents"]].to_numpy(dtype=float)
        if kernel["type"] == "lg":
            means = kernel["beta0"] + pa @ kernel["betas"]
            return np.zeros((len(state), 1)), means[:, None], kernel["var"]

        means = kernel["offset"][None, :] + (pa @ kernel["reg"])[:, None]
        if kernel["whiten"] is None:
            log_w = np.zeros_like(means)
        else:
            z = pa @ kernel["whiten"]
            sq_dist = (
                np.sum(z**2, axis=1)[:, None]
                + kernel["z_train_sq"][None, :]
                - 2.0 * z @ kernel["z_train"].T
            )
            log_w = -0.5 * np.maximum(sq_dist, 0.0)
        return log_w, means, kernel["var"]

    def _lg_children_gaussian(
        self, node: str, state: pd.DataFrame, lg_children: list[str]
    ) -> tuple[np.ndarray, float] | None:
        """Product of the LinearGaussian children likelihoods as a function of ``node``.
        Each child c ~ N(beta0 + beta_x x + beta_o·o, var) is Gaussian in x, so the product is N(x; mu, tau2) up to a constant.
        Returns None if no child depends on ``node``."""
        precision = 0.0
        weighted_sum = np.zeros(len(state))
        for child in lg_children:
            cpd = self.cpd(child)
            parents = cpd.evidence()
            beta = np.asarray(cpd.beta, dtype=float)
            node_index = parents.index(node)
            beta_x = beta[node_index + 1]
            if beta_x == 0.0:
                continue
            others = [p for p in parents if p != node]
            other_betas = np.delete(beta[1:], node_index)
            rest = beta[0] + state[others].to_numpy(dtype=float) @ other_betas
            residual = state[child].to_numpy(dtype=float) - rest
            precision += beta_x**2 / cpd.variance
            weighted_sum += beta_x * residual / cpd.variance
        if precision == 0.0:
            return None
        tau2 = 1.0 / precision
        return tau2 * weighted_sum, tau2

    @staticmethod
    def _sample_mixture(
        log_w: np.ndarray, means: np.ndarray, var: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Draws one value per row from a Gaussian mixture (component chosen with the Gumbel-max trick)"""
        component = np.argmax(log_w + rng.gumbel(size=log_w.shape), axis=1)
        mean = means[np.arange(len(means)), component]
        return mean + np.sqrt(var) * rng.standard_normal(len(means))

    def _gibbs_update(
        self,
        node: str,
        state: pd.DataFrame,
        kernels: dict[str, dict],
        lg_children: list[str],
        ckde_children: list[str],
        rng: np.random.Generator,
        max_rejection_tries: int,
    ) -> tuple[np.ndarray, int, int]:
        """Samples ``node`` from its full conditional p(x | Markov blanket) for every chain.

        q(x) = p(x | pa) * prod_{LG children} p(c | .) is a Gaussian mixture sampled exactly. CKDE children
        g(x) = prod_{CKDE children} p(c | .) are handled by rejection sampling with the bound g <= B, and the chains
        still pending after ``max_rejection_tries`` rounds take one Metropolis-Hastings step with proposal q.

        Returns:
            tuple[np.ndarray, int, int]: The new values, the number of exact draws and the number of changed values.
        """
        log_w, means, var = self._conditional_mixture(kernels[node], state)
        lg_gaussian = self._lg_children_gaussian(node, state, lg_children)
        if lg_gaussian is not None:
            mu_lg, tau2 = lg_gaussian
            # N(x; m_j, var) N(x; mu_lg, tau2) = N(m_j; mu_lg, var + tau2) N(x; m*_j, v*)
            log_w = log_w + norm.logpdf(
                means, loc=mu_lg[:, None], scale=np.sqrt(var + tau2)
            )
            post_var = 1.0 / (1.0 / var + 1.0 / tau2)
            means = post_var * (means / var + mu_lg[:, None] / tau2)
            var = post_var

        n_chains = len(state)
        if len(ckde_children) == 0:
            return self._sample_mixture(log_w, means, var, rng), n_chains, n_chains

        def _log_g(rows: np.ndarray, values: np.ndarray) -> np.ndarray:
            proposed_state = state.loc[rows].copy()
            proposed_state[node] = values
            log_g = sum(self._node_logl(c, proposed_state) for c in ckde_children)
            return np.nan_to_num(log_g, nan=-np.inf)

        log_bound = sum(kernels[c]["log_bound"] for c in ckde_children)
        new_values = state[node].to_numpy(dtype=float).copy()
        pending = np.ones(n_chains, dtype=bool)
        for _ in range(max_rejection_tries):
            rows = np.flatnonzero(pending)
            if len(rows) == 0:
                break
            proposal = self._sample_mixture(log_w[rows], means[rows], var, rng)
            accept = np.log(rng.random(len(rows))) < _log_g(rows, proposal) - log_bound
            new_values[rows[accept]] = proposal[accept]
            pending[rows[accept]] = False
        n_exact = n_chains - int(pending.sum())
        n_changed = n_exact

        # MH fallback with the same proposal q: alpha = min(1, g(x*) / g(x))
        rows = np.flatnonzero(pending)
        if len(rows) > 0:
            proposal = self._sample_mixture(log_w[rows], means[rows], var, rng)
            log_alpha = _log_g(rows, proposal) - _log_g(rows, new_values[rows])
            accept = np.log(rng.random(len(rows))) < log_alpha
            new_values[rows[accept]] = proposal[accept]
            n_changed += int(accept.sum())
        return new_values, n_exact, n_changed

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
        max_rejection_tries: int = 10,
    ) -> dict[str, dict]:
        """
        Performs Gibbs sampling inference on the Bayesian network using the provided evidence.
        Each unobserved node X is sampled from its full conditional p(x | MB(x)) ∝ p(x | Pa(x)) prod_c p(c | Pa(c)):
            - p(x | Pa(x)) is a Gaussian mixture (CKDE kernels, or one component for LinearGaussian CPDs), and multiplying it by the LinearGaussian children (Gaussian in x) keeps it a Gaussian mixture, which is sampled exactly.
            - CKDE children have no closed form, so they are handled by rejection sampling using the bound p(c | .) <= 1 / sqrt(2 pi s2_c).
            Chains not accepted after ``max_rejection_tries`` rounds take one Metropolis-Hastings step with the same proposal, which still leaves the full conditional invariant.
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
            max_rejection_tries (int, optional): Maximum rejection sampling rounds per update before falling back to Metropolis-Hastings. Defaults to 10.
        Returns:
            dict[str, dict]: A dictionary containing the structure of the Bayesian network and the parameters of the inference results. The parameters include the (uniform) weights, the assignments and, per node, the fraction of exact draws from the full conditional (``exact_rate``) and the fraction of updates that changed the value (``acceptance_rate``).
        """
        rng = np.random.default_rng(seed)

        def _next_seed() -> int:
            return int(rng.integers(2**31 - 1))

        topo_order = list(self.graph().topological_sort())
        free_nodes = [node for node in topo_order if node not in evidence]
        kernels = {node: self._kernel_cache(node) for node in topo_order}
        lg_children = {
            node: [c for c in self.children(node) if kernels[c]["type"] == "lg"]
            for node in free_nodes
        }
        ckde_children = {
            node: [c for c in self.children(node) if kernels[c]["type"] == "ckde"]
            for node in free_nodes
        }
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
        exact = {node: 0 for node in free_nodes}
        changed = {node: 0 for node in free_nodes}
        kept_states = []

        for sweep in range(n_sweeps):
            for node in free_nodes:
                new_values, n_exact, n_changed = self._gibbs_update(
                    node,
                    state,
                    kernels,
                    lg_children[node],
                    ckde_children[node],
                    rng,
                    max_rejection_tries,
                )
                state[node] = new_values
                exact[node] += n_exact
                changed[node] += n_changed

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
            "exact_rate": {
                node: exact[node] / (n_sweeps * n_chains) for node in free_nodes
            },
            "acceptance_rate": {
                node: changed[node] / (n_sweeps * n_chains) for node in free_nodes
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
