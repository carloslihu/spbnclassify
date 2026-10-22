import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pybnesian as pbn
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import logsumexp
from scipy.stats import norm, rankdata

from ..utils.constants import TRUE_ANOMALY_LABEL
from .base import BayesianNetwork

# Thresholds of the inference diagnostics below which a RuntimeWarning is emitted
MIN_EFFECTIVE_SAMPLE_SIZE = 100
MAX_RHAT = 1.05


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

    def _clg_conditional(self, sampled_nodes: set[str]) -> dict:
        """Computes the Gaussian distribution of the non-sampled LinearGaussian nodes R given the sampled nodes S.
        Since S is ancestor-closed, R = c + B_S S + B_R R + eps with eps ~ N(0, diag(var)), so with A = (I - B_R)^-1
        R | S ~ N(A c + A B_S S, A diag(var) A^T): the mean is affine in S and the covariance does not depend on S.

        Returns:
            dict: The R nodes, the S nodes, the intercept A c, the coefficients A B_S and the covariance of R | S.
        """
        topo_order = list(self.graph().topological_sort())
        rb_nodes = [
            node
            for node in topo_order
            if node not in sampled_nodes
            and self.cpd(node).type() == pbn.LinearGaussianCPDType()
        ]
        conditioning_nodes = [node for node in topo_order if node in sampled_nodes]
        rb_index = {node: i for i, node in enumerate(rb_nodes)}
        conditioning_index = {node: i for i, node in enumerate(conditioning_nodes)}

        intercept = np.zeros(len(rb_nodes))
        variance = np.zeros(len(rb_nodes))
        B_S = np.zeros((len(rb_nodes), len(conditioning_nodes)))
        B_R = np.zeros((len(rb_nodes), len(rb_nodes)))
        for i, node in enumerate(rb_nodes):
            cpd = self.cpd(node)
            beta = np.asarray(cpd.beta, dtype=float)
            intercept[i] = beta[0]
            variance[i] = cpd.variance
            for parent, coef in zip(cpd.evidence(), beta[1:]):
                if parent in rb_index:
                    B_R[i, rb_index[parent]] = coef
                else:
                    B_S[i, conditioning_index[parent]] = coef

        # B_R is strictly lower triangular in topological order, so I - B_R is invertible
        A = np.linalg.inv(np.eye(len(rb_nodes)) - B_R)
        return {
            "nodes": rb_nodes,
            "conditioning_nodes": conditioning_nodes,
            "intercept": A @ intercept,
            "coefficients": A @ B_S,
            "covariance": (A * variance) @ A.T,
        }

    def _rao_blackwellize_clg(
        self,
        assignments: pd.DataFrame,
        evidence: dict[str, float],
        sampled_nodes: set[str],
    ) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
        """Rao-Blackwellizes the non-sampled LinearGaussian nodes R given the sampled values S of each row (see ``_clg_conditional``).
        With E the evidence nodes of R and Q the rest, the Gaussian conditioning gives in closed form:
            - log p(e_E | S), the evidence log-likelihood marginalizing Q.
            - Q | S, e_E ~ N(m_Q + K (e_E - m_E), Sigma_QQ - K Sigma_EQ) with K = Sigma_QE Sigma_EE^-1, whose covariance does not depend on S.

        Args:
            assignments (pd.DataFrame): Sampled assignments (one row per sample).
            evidence (dict[str, float]): Observed variables.
            sampled_nodes (set[str]): Ancestor-closed set of nodes whose sampled values are conditioned on. It must contain all the CKDE nodes.

        Returns:
            tuple[np.ndarray, pd.DataFrame, dict[str, float]]: The row-wise log p(e_E | S), the assignments with the posterior means
            in the Q columns (and the evidence in the E columns), and the posterior std of each Q node.
        """
        log_lik = np.zeros(len(assignments), dtype=float)
        clg = self._clg_conditional(sampled_nodes)
        rb_nodes = clg["nodes"]
        if len(rb_nodes) == 0:
            return log_lik, assignments, {}

        conditioning_values = assignments[clg["conditioning_nodes"]].to_numpy(
            dtype=float
        )
        means = clg["intercept"][None, :] + conditioning_values @ clg["coefficients"].T
        covariance = clg["covariance"]
        evidence_index = [i for i, node in enumerate(rb_nodes) if node in evidence]
        query_index = [i for i, node in enumerate(rb_nodes) if node not in evidence]

        query_means = means[:, query_index]
        query_covariance = covariance[np.ix_(query_index, query_index)]
        if len(evidence_index) > 0:
            evidence_values = np.array(
                [float(evidence[rb_nodes[i]]) for i in evidence_index]
            )
            residuals = evidence_values[None, :] - means[:, evidence_index]
            chol = cholesky(
                covariance[np.ix_(evidence_index, evidence_index)], lower=True
            )
            whitened = solve_triangular(chol, residuals.T, lower=True)
            log_lik = (
                -0.5 * np.sum(whitened**2, axis=0)
                - np.sum(np.log(np.diag(chol)))
                - 0.5 * len(evidence_index) * np.log(2.0 * np.pi)
            )
            # gain = K^T = Sigma_EE^-1 Sigma_EQ
            cross_covariance = covariance[np.ix_(evidence_index, query_index)]
            gain = cho_solve((chol, True), cross_covariance)
            query_means = query_means + residuals @ gain
            query_covariance = query_covariance - cross_covariance.T @ gain

        for node in rb_nodes:
            if node in evidence:
                assignments[node] = float(evidence[node])
        posterior_std = {}
        for k, i in enumerate(query_index):
            assignments[rb_nodes[i]] = query_means[:, k]
            posterior_std[rb_nodes[i]] = float(np.sqrt(query_covariance[k, k]))
        return log_lik, assignments, posterior_std

    def _export_inference(
        self,
        infer_dict: dict[str, dict],
        json_file_path: Path | None = None,
        pdf_file_path: Path | None = None,
    ) -> None:
        """Exports the inference results to JSON and/or the network graph to PDF"""
        if json_file_path:
            # Builds new dicts so the returned infer_dict is not modified
            export_dict = {
                **infer_dict,
                "parameters": {
                    **infer_dict["parameters"],
                    "weights": infer_dict["parameters"]["weights"].tolist(),
                    "assignments": infer_dict["parameters"]["assignments"].to_dict(
                        orient="list"
                    ),
                },
            }
            with open(json_file_path, "w") as f:
                json.dump(self._json_safe(export_dict), f, indent=4)
        if pdf_file_path:
            self.save(pdf_file_path)

    @classmethod
    def _json_safe(cls, value):
        """Recursively replaces the non-finite floats (e.g., NaN diagnostics) by None, which strict JSON parsers accept as null"""
        if isinstance(value, dict):
            return {key: cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _warn_diagnostics(
        rhat: dict[str, float], effective_sample_size: dict[str, float]
    ) -> None:
        """Emits a RuntimeWarning for the nodes whose R-hat exceeds ``MAX_RHAT`` or whose effective sample size is below ``MIN_EFFECTIVE_SAMPLE_SIZE``"""
        high_rhat = {node: value for node, value in rhat.items() if value > MAX_RHAT}
        if high_rhat:
            warnings.warn(
                f"R-hat above {MAX_RHAT} for {high_rhat}: the chains have not converged or are too short for their "
                "autocorrelation, increase burn_in or the draws per chain (n_samples / n_chains).",
                RuntimeWarning,
                stacklevel=3,
            )
        low_ess = {
            node: value
            for node, value in effective_sample_size.items()
            if value < MIN_EFFECTIVE_SAMPLE_SIZE
        }
        if low_ess:
            warnings.warn(
                f"Effective sample size below {MIN_EFFECTIVE_SAMPLE_SIZE} for {low_ess}: the estimates are unreliable, "
                "increase n_samples.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _posterior_density(
        self,
        query_node: str,
        point: pd.Series,
        assignments: pd.DataFrame,
        weights: np.ndarray,
        posterior_std: dict[str, float],
        effective_sample_size: float | None = None,
    ) -> float:
        """Estimates the posterior density of ``query_node`` at ``point`` from weighted samples.
        Sampled columns use a weighted Gaussian KDE, Rao-Blackwellized CLG columns (the keys of ``posterior_std``)
        a weighted mixture of their CLG posteriors N(posterior mean of the row, posterior std).
        ``effective_sample_size`` overrides the importance-sampling n_eff = 1 / sum(w**2) of the bandwidth (e.g., the autocorrelation-aware ESS of MCMC samples).
        """

        def _kernel_values(samples: np.ndarray) -> np.ndarray:
            # Weighted Silverman’s rule of thumb: 1.06 * std_w * n_eff**(-1/5), with the effective sample size n_eff = 1 / sum(w**2) by default
            weighted_mean = np.sum(weights * samples)
            weighted_std = np.sqrt(np.sum(weights * (samples - weighted_mean) ** 2))
            n_eff = (
                effective_sample_size
                if effective_sample_size is not None
                and np.isfinite(effective_sample_size)
                else 1.0 / np.sum(weights**2)
            )
            bandwidth = 1.06 * weighted_std * (n_eff ** (-1.0 / 5.0))
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                bandwidth = max(weighted_std, 1.0)

            # Computes Gaussian kernel values at the target point[query_node]
            normalized_deltas = (float(point[query_node]) - samples) / bandwidth
            kernel_values = np.exp(-0.5 * normalized_deltas**2) / (
                np.sqrt(2.0 * np.pi) * bandwidth
            )

            return kernel_values

        if query_node in posterior_std:
            # Rao-Blackwellized CLG nodes: the rows hold the posterior means, and the posterior std is shared by all rows
            kernel_values = norm.pdf(
                float(point[query_node]),
                loc=assignments[query_node].to_numpy(dtype=float),
                scale=posterior_std[query_node],
            )
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
        Performs likelihood weighting inference on the Bayesian network using the provided evidence.
        It is Rao-Blackwellized: only the CKDE nodes and their LinearGaussian ancestors are sampled, and the remaining LinearGaussian nodes are replaced by their exact CLG posterior.
        The weights include the evidence likelihood of the sampled nodes and the marginal likelihood of the evidence on the remaining LinearGaussian nodes.
        The assignments of the Rao-Blackwellized nodes hold their posterior mean per row, and ``posterior_std`` their posterior std (the same for all rows).
        Args:
            evidence (dict[str, float], optional): A dictionary mapping node names to their observed values (hard evidence only), e.g. {"A": 1.2, "D": -0.4}. Defaults to an empty dictionary.
            n_samples (int, optional): The number of samples to draw for the likelihood weighting inference. Defaults to 1,000.
            seed (int, optional): The random seed for reproducibility. Defaults to 0.
            json_file_path (Path | None, optional): If provided, exports the inference results to this file in JSON format.
            pdf_file_path (Path | None, optional): If provided, exports the graphical representation of the inference to this file in PDF format.
        Returns:
            dict[str, dict]: A dictionary containing the structure of the Bayesian network (list of arcs), the evidence and the parameters of the inference results.
            The parameters include the normalized weights, the assignments, the posterior std of the Rao-Blackwellized nodes (``posterior_std``) and
            the effective sample size n_eff = 1 / sum(w**2) (``effective_sample_size``). A small n_eff compared to ``n_samples`` means the weights are
            degenerate (e.g., unlikely evidence on the sampled nodes), so the estimates are unreliable and more samples are needed:
            a RuntimeWarning is emitted below ``MIN_EFFECTIVE_SAMPLE_SIZE``.
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
        log_rb_evidence, assignments, posterior_std = self._rao_blackwellize_clg(
            assignments, evidence, sampled
        )
        log_weights += log_rb_evidence

        # Normalize log weights to get importance weights.
        weights = np.exp(log_weights - logsumexp(log_weights))
        effective_sample_size = float(1.0 / np.sum(weights**2))
        # Without sampled nodes every row is the exact posterior, so the sample size does not matter
        if len(sampled) > 0 and effective_sample_size < MIN_EFFECTIVE_SAMPLE_SIZE:
            warnings.warn(
                f"Effective sample size {effective_sample_size:.1f} below {MIN_EFFECTIVE_SAMPLE_SIZE}: the weights are "
                "degenerate (e.g., unlikely evidence on the sampled nodes), so the estimates are unreliable, increase n_samples.",
                RuntimeWarning,
                stacklevel=2,
            )
        infer_dict["parameters"] = {
            "weights": weights,
            "assignments": assignments,
            "posterior_std": posterior_std,
            "effective_sample_size": effective_sample_size,
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
            The number of samples to draw for the likelihood weighting inference. Defaults to 1,000.
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
            likelihood_weighting_dict["parameters"]["posterior_std"],
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

    def _rb_evidence_model(
        self, sampled_nodes: set[str], evidence: dict[str, float]
    ) -> dict | None:
        """Precomputes p(e_E | S) = N(e_E; c_E + C_E s, Sigma_EE) of the evidence E on the non-sampled LinearGaussian nodes
        given the sampled values s (see ``_clg_conditional``). Returns None if there is no such evidence.
        """
        clg = self._clg_conditional(sampled_nodes)
        evidence_index = [i for i, node in enumerate(clg["nodes"]) if node in evidence]
        if len(evidence_index) == 0:
            return None
        return {
            "conditioning_nodes": clg["conditioning_nodes"],
            "evidence_values": np.array(
                [float(evidence[clg["nodes"][i]]) for i in evidence_index]
            ),
            "intercept": clg["intercept"][evidence_index],
            "coefficients": clg["coefficients"][evidence_index],
            "precision": np.linalg.inv(
                clg["covariance"][np.ix_(evidence_index, evidence_index)]
            ),
        }

    @staticmethod
    def _rb_evidence_gaussian(
        node: str, state: pd.DataFrame, rb_evidence: dict
    ) -> tuple[np.ndarray, float] | None:
        """p(e_E | S) as a function of the sampled ``node`` x = s_j. With a = C_E[:, j], P = Sigma_EE^-1 and the residual
        r = e_E - c_E - C_E s + a x (which does not depend on x), it is N(x; a^T P r / a^T P a, 1 / a^T P a) up to a constant.
        Returns None if the evidence does not depend on ``node``."""
        a = rb_evidence["coefficients"][
            :, rb_evidence["conditioning_nodes"].index(node)
        ]
        if not np.any(a):
            return None
        precision_a = rb_evidence["precision"] @ a
        tau2 = 1.0 / (a @ precision_a)
        conditioning_values = state[rb_evidence["conditioning_nodes"]].to_numpy(
            dtype=float
        )
        residuals = (
            rb_evidence["evidence_values"][None, :]
            - rb_evidence["intercept"][None, :]
            - conditioning_values @ rb_evidence["coefficients"].T
            + state[node].to_numpy(dtype=float)[:, None] * a[None, :]
        )
        return tau2 * (residuals @ precision_a), tau2

    @staticmethod
    def _gaussian_product(
        first: tuple[np.ndarray, float] | None,
        second: tuple[np.ndarray, float] | None,
    ) -> tuple[np.ndarray, float] | None:
        """N(x; mu1, tau1) N(x; mu2, tau2) is N(x; mu, tau) up to a constant, with 1 / tau = 1 / tau1 + 1 / tau2 and
        mu = tau (mu1 / tau1 + mu2 / tau2). A None factor is ignored."""
        if first is None:
            return second
        if second is None:
            return first
        (mu1, tau1), (mu2, tau2) = first, second
        tau = 1.0 / (1.0 / tau1 + 1.0 / tau2)
        return tau * (mu1 / tau1 + mu2 / tau2), tau

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
        rb_evidence: dict | None = None,
    ) -> tuple[np.ndarray, int, int]:
        """Samples ``node`` from its full conditional p(x | Markov blanket) for every chain.

        q(x) = p(x | pa) * prod_{LG children} p(c | .) * p(e_E | S) is a Gaussian mixture sampled exactly, where p(e_E | S)
        is the evidence likelihood of the integrated-out LinearGaussian nodes (see ``_rb_evidence_gaussian``). CKDE children
        g(x) = prod_{CKDE children} p(c | .) are handled by rejection sampling with the bound g <= B, and the chains
        still pending after ``max_rejection_tries`` rounds take one Metropolis-Hastings step with proposal q.

        Returns:
            tuple[np.ndarray, int, int]: The new values, the number of exact draws and the number of changed values.
        """
        log_w, means, var = self._conditional_mixture(kernels[node], state)
        lg_gaussian = self._gaussian_product(
            self._lg_children_gaussian(node, state, lg_children),
            (
                self._rb_evidence_gaussian(node, state, rb_evidence)
                if rb_evidence is not None
                else None
            ),
        )
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

    @staticmethod
    def _split_chains(draws: np.ndarray) -> np.ndarray:
        """Splits each chain of ``draws`` (n_draws, n_chains) into its first and last halves (dropping the middle draw if n_draws
        is odd), so that within-chain trends show up as between-chain differences. Returns (n_draws // 2, 2 * n_chains).
        """
        half = draws.shape[0] // 2
        return np.concatenate([draws[:half], draws[draws.shape[0] - half :]], axis=1)

    @staticmethod
    def _rank_normalize(draws: np.ndarray) -> np.ndarray:
        """Rank-normalizes the pooled ``draws``: z = Phi^-1((rank - 3/8) / (n + 1/4)) with average ranks for ties (Vehtari et al., 2021)"""
        ranks = rankdata(draws, method="average").reshape(draws.shape)
        return norm.ppf((ranks - 3.0 / 8.0) / (draws.size + 1.0 / 4.0))

    @staticmethod
    def _rhat_raw(chains: np.ndarray) -> float:
        """R-hat of already split ``chains`` (n, m): sqrt(var+ / W), with W the mean within-chain variance and
        var+ = (n - 1) / n W + B / n the pooled variance. Returns NaN for constant draws.
        """
        n = chains.shape[0]
        within = np.mean(np.var(chains, axis=0, ddof=1))
        if within == 0.0:
            return np.nan
        between_over_n = np.var(np.mean(chains, axis=0), ddof=1)
        return float(np.sqrt(((n - 1) / n * within + between_over_n) / within))

    @classmethod
    def _split_rhat(cls, draws: np.ndarray) -> float:
        """Rank-normalized split-R-hat (Vehtari et al., 2021) of ``draws`` (n_draws, n_chains): the maximum of the split-R-hat
        of the rank-normalized draws (bulk) and of the rank-normalized folded draws |x - median| (tails). Values close to 1
        (e.g., < 1.01-1.05) indicate convergence. Returns NaN with fewer than 4 draws or 2 chains, or constant draws.
        """
        if draws.shape[0] < 4 or draws.shape[1] < 2:
            return np.nan
        chains = cls._split_chains(draws)
        rhat_bulk = cls._rhat_raw(cls._rank_normalize(chains))
        rhat_tail = cls._rhat_raw(
            cls._rank_normalize(np.abs(chains - np.median(chains)))
        )
        return float(max(rhat_bulk, rhat_tail))

    @staticmethod
    def _ess_raw(chains: np.ndarray) -> float:
        """Autocorrelation-aware effective sample size (Stan/BDA3) of already split ``chains`` (n, m): n m / tau, with the
        integrated autocorrelation time tau = -1 + 2 sum_t rho_t truncated with Geyer's initial monotone positive sequence.
        """
        n, m = chains.shape
        n_total = n * m
        if np.max(chains) - np.min(chains) < np.finfo(float).resolution:
            return float(n_total)
        # Biased autocovariance of each chain at every lag via FFT
        centered = chains - np.mean(chains, axis=0)
        n_fft = 2 ** int(np.ceil(np.log2(2 * n)))
        spectrum = np.fft.rfft(centered, n=n_fft, axis=0)
        acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft, axis=0)[:n] / n
        mean_acov = np.mean(acov, axis=1)

        within = mean_acov[0] * n / (n - 1)
        var_plus = within * (n - 1) / n
        if m > 1:
            var_plus += np.var(np.mean(chains, axis=0), ddof=1)
        rho = np.zeros(n)
        rho[0] = 1.0
        rho_even = 1.0
        rho_odd = 1.0 - (within - mean_acov[1]) / var_plus
        rho[1] = rho_odd
        # Geyer's initial positive sequence: sum pairs of autocorrelations while they are positive
        t = 1
        while t < n - 3 and rho_even + rho_odd > 0.0:
            rho_even = 1.0 - (within - mean_acov[t + 1]) / var_plus
            rho_odd = 1.0 - (within - mean_acov[t + 2]) / var_plus
            if rho_even + rho_odd >= 0.0:
                rho[t + 1] = rho_even
                rho[t + 2] = rho_odd
            t += 2
        max_t = t - 2
        if rho_even > 0.0:
            rho[max_t + 1] = rho_even
        # Geyer's initial monotone sequence: the pair sums must be non-increasing
        t = 1
        while t <= max_t - 2:
            if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
                rho[t + 1] = rho[t + 2] = (rho[t - 1] + rho[t]) / 2.0
            t += 2
        tau = -1.0 + 2.0 * np.sum(rho[: max_t + 1]) + np.sum(rho[max_t + 1 : max_t + 2])
        tau = max(tau, 1.0 / np.log10(n_total))
        return float(n_total / tau)

    @classmethod
    def _mcmc_effective_sample_size(cls, draws: np.ndarray) -> float:
        """Bulk effective sample size (Vehtari et al., 2021) of ``draws`` (n_draws, n_chains): the autocorrelation-aware ESS
        of the rank-normalized split chains. Returns NaN with fewer than 4 draws per chain.
        """
        if draws.shape[0] < 4:
            return np.nan
        return cls._ess_raw(cls._rank_normalize(cls._split_chains(draws)))

    @classmethod
    def _tail_effective_sample_size(cls, draws: np.ndarray) -> float:
        """Tail effective sample size (Vehtari et al., 2021) of ``draws`` (n_draws, n_chains): the minimum ESS of the indicators
        I(x <= q_0.05) and I(x <= q_0.95), i.e. of the 5% and 95% quantile estimates. Returns NaN with fewer than 4 draws per chain.
        """
        if draws.shape[0] < 4:
            return np.nan
        return min(
            cls._ess_raw(
                cls._split_chains((draws <= np.quantile(draws, prob)).astype(float))
            )
            for prob in (0.05, 0.95)
        )

    # RFE: Use the convergence diagnostics (R-hat) as automatic stop criterion
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
        Performs collapsed (Rao-Blackwellized) Gibbs sampling inference on the Bayesian network using the provided evidence.
        As in ``infer``, only the CKDE nodes and their LinearGaussian ancestors S are sampled: the remaining LinearGaussian nodes R
        are integrated out, so the chains target p(S | e) ∝ p(S, e_S) p(e_R | S), and R is finally replaced by its exact CLG posterior p(R | S, e).
        Each unobserved node X of S is sampled from its full conditional p(x | .) ∝ p(x | Pa(x)) prod_{c in S} p(c | Pa(c)) p(e_R | S):
            - p(x | Pa(x)) is a Gaussian mixture (CKDE kernels, or one component for LinearGaussian CPDs), and multiplying it by the LinearGaussian children in S and by p(e_R | S) (both Gaussian in x) keeps it a Gaussian mixture, which is sampled exactly.
            - CKDE children have no closed form, so they are handled by rejection sampling using the bound p(c | .) <= 1 / sqrt(2 pi s2_c).
            Chains not accepted after ``max_rejection_tries`` rounds take one Metropolis-Hastings step with the same proposal, which still leaves the full conditional invariant.
        ``n_chains`` independent chains are run in batch. Integrating R out avoids the slow mixing of Gibbs sampling between strongly dependent nodes of S and R.
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
            dict[str, dict]: A dictionary containing the structure of the Bayesian network, the evidence and the parameters of the inference results. The parameters include the (uniform) weights, the assignments, the posterior std of the Rao-Blackwellized nodes (``posterior_std``) and:
                - ``exact_rate`` (per sampled node): the fraction of exact draws from the full conditional.
                - ``acceptance_rate`` (per sampled node): the fraction of updates that changed the value.
                - ``rhat`` (per non-evidence node): the rank-normalized split-R-hat over the chains of the returned values (posterior means for the Rao-Blackwellized
                  nodes). Values above ~1.01-1.05 mean that the chains have not converged or are too short for their autocorrelation (increase ``burn_in`` or the
                  draws per chain, i.e. ``n_samples / n_chains``).
                - ``effective_sample_size`` (per non-evidence node): the bulk effective sample size of the returned values. Much smaller than ``n_samples`` means
                  highly autocorrelated chains (increase ``n_samples`` or ``thinning``).
                - ``tail_effective_sample_size`` (per non-evidence node): the effective sample size of the 5% and 95% quantile estimates.
            The diagnostics are NaN with fewer than 4 kept sweeps per chain (and R-hat with fewer than 2 chains or constant values).
            A RuntimeWarning is emitted if any R-hat exceeds ``MAX_RHAT`` or any bulk effective sample size is below ``MIN_EFFECTIVE_SAMPLE_SIZE``.
        """
        rng = np.random.default_rng(seed)

        def _next_seed() -> int:
            return int(rng.integers(2**31 - 1))

        topo_order = list(self.graph().topological_sort())
        # Only the ancestor-closed set S of CKDE nodes and their LinearGaussian ancestors is sampled, the rest is integrated out.
        sampled = self._sampled_nodes()
        sampled_order = [node for node in topo_order if node in sampled]
        free_nodes = [node for node in sampled_order if node not in evidence]
        kernels = {node: self._kernel_cache(node) for node in sampled_order}
        # Children outside S are integrated out, and their evidence enters through p(e_R | S)
        lg_children = {
            node: [
                c
                for c in self.children(node)
                if c in sampled and kernels[c]["type"] == "lg"
            ]
            for node in free_nodes
        }
        ckde_children = {
            node: [
                c
                for c in self.children(node)
                if c in sampled and kernels[c]["type"] == "ckde"
            ]
            for node in free_nodes
        }
        rb_evidence = self._rb_evidence_model(sampled, evidence)
        infer_dict = {
            "structure": self.arcs(),
            "evidence": dict(evidence),
            "parameters": {},
        }

        # Initialize the chains by forward sampling S (ancestor-closed) with the evidence clamped.
        state = pd.DataFrame(
            index=np.arange(n_chains), columns=sampled_order, dtype=float
        )
        for node in sampled_order:
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
                    rb_evidence,
                )
                state[node] = new_values
                exact[node] += n_exact
                changed[node] += n_changed

            if sweep >= burn_in and (sweep - burn_in) % thinning == thinning - 1:
                kept_states.append(state.copy())

        # The chains sample p(S | e), so the Rao-Blackwellized nodes only need p(R | S, e) and the evidence likelihood is not a weight.
        all_assignments = pd.DataFrame(
            np.concatenate([kept.to_numpy(dtype=float) for kept in kept_states]),
            columns=sampled_order,
        )
        _, all_assignments, posterior_std = self._rao_blackwellize_clg(
            all_assignments, evidence, sampled
        )
        all_assignments = all_assignments[topo_order]

        # Convergence diagnostics of the returned values (posterior means for the Rao-Blackwellized nodes) on all the kept
        # sweeps, whose rows are ordered sweep by sweep, i.e. (n_kept_sweeps, n_chains) per node
        n_kept = len(kept_states)
        query_nodes = [node for node in topo_order if node not in evidence]
        draws = {
            node: all_assignments[node].to_numpy(dtype=float).reshape(n_kept, n_chains)
            for node in query_nodes
        }
        assignments = all_assignments.iloc[:n_samples].reset_index(drop=True)
        # Only the first n_samples draws are returned, so the effective sample sizes are scaled accordingly
        kept_fraction = len(assignments) / len(all_assignments)
        rhat = {node: self._split_rhat(draws[node]) for node in query_nodes}
        effective_sample_size = {
            node: kept_fraction * self._mcmc_effective_sample_size(draws[node])
            for node in query_nodes
        }
        tail_effective_sample_size = {
            node: kept_fraction * self._tail_effective_sample_size(draws[node])
            for node in query_nodes
        }
        # Without free sampled nodes every row is the exact posterior, so there is nothing to diagnose
        if len(free_nodes) > 0:
            self._warn_diagnostics(rhat, effective_sample_size)

        # MCMC samples are equally weighted.
        weights = np.full(len(assignments), 1.0 / len(assignments))
        infer_dict["parameters"] = {
            "weights": weights,
            "assignments": assignments,
            "posterior_std": posterior_std,
            "exact_rate": {
                node: exact[node] / (n_sweeps * n_chains) for node in free_nodes
            },
            "acceptance_rate": {
                node: changed[node] / (n_sweeps * n_chains) for node in free_nodes
            },
            "rhat": rhat,
            "effective_sample_size": effective_sample_size,
            "tail_effective_sample_size": tail_effective_sample_size,
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

        Raises
        ------
        ValueError
            If the query node is in the evidence, or if ``gibbs_sampling_dict`` was computed with a different evidence.
        """
        if query_node in evidence:
            raise ValueError("Query node and evidence nodes must be disjoint.")
        if (
            gibbs_sampling_dict != {}
            and gibbs_sampling_dict.get("evidence") != evidence
        ):
            raise ValueError(
                "gibbs_sampling_dict was computed with a different evidence."
            )
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
            gibbs_sampling_dict["parameters"]["posterior_std"],
            # The KDE bandwidth uses the autocorrelation-aware effective sample size of the query node
            gibbs_sampling_dict["parameters"]
            .get("effective_sample_size", {})
            .get(query_node),
        )
