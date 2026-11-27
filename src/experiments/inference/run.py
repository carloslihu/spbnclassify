"""Experiments of the article on Rao-Blackwellized inference in semiparametric Bayesian networks (sections/04-results.tex).

Each experiment (E1-E7) writes its raw results to ``data/results/inference/<experiment>/``:
    - ``units/<unit_id>/``: the files of each independent unit (a network or a dataset),
    written as its stages finish, so an interrupted run resumes by skipping the finished units:
        - ``network.pkl``: the fitted network (or classifier), which the timed stage loads.
        - ``network.json``: its description, with the timing device added by the timed stage.
        - ``train_features.csv`` and ``test_rows.csv`` (E7): the training features of the imputers and the test instances.
        - ``tasks.jsonl``, ``references.jsonl`` and ``results.csv``: as below, for this unit only.
        - ``results_gpu.csv`` and ``results_cpu.csv``: the runs of the LW family (GPU workers) and of the Gibbs family and
          imputation (CPU workers), the two parts of the timed stage, concatenated in ``results.csv``.
        - ``timing_gpu.json`` and ``timing_cpu.json``: the GPU and cores of the worker of each part.
        - ``references_done``, ``done_gpu``, ``done_cpu`` and ``done``: empty markers of the finished reference stage,
          parts of the timed stage, and unit.
    - ``results.csv``, ``tasks.jsonl``, ``references.jsonl`` and ``networks.csv``: the concatenation of all the units.
        - ``networks.csv``: one row per unit, with the realized structure (``n_nodes``, ``n_ckde``, ``n_clg``, ``n_rb``,
          ``frac_rb``, ``n_delta``, ``mean_ckde_size``), the generator settings (``config_*``), ``network_seed``, the
          factor and level, and the GPU and cores of the timed runs.
        - ``tasks.jsonl``: one record per task, with the query, its type, the evidence values, the task factors
          (evidence location, rarity, k, ...) and the true class (E7 only).
        - ``references.jsonl``: one record per task with the reference posterior: its method (exact, grid or lw),
          whether it passed the check (``ok``, and ``message`` otherwise), mean, std, class probabilities, the CDF on a
          1024-point grid (for W1), and its MCSE and effective sample size.
        - ``results.csv``: one row per (task, method, N, repetition), with the estimates, errors, cost and diagnostics of
          the run.
``analysis.py`` builds the figures, tables and statistical tests from these files.

Usage (from the library root):
    python -m src.experiments.inference.run E2 [--quick] [--max_workers 1] [--cpu_workers 15] [--n_reps 20] ...
"""

# region Imports
import os

# The OpenCL device must be chosen before pybnesian is imported, and the BLAS threads fixed before numpy is imported
from ...utils.opencl_device import (
    assign_worker_device,
    available_gpus,
    opencl_device_name,
    select_most_available_gpu,
    worker_slots,
)

select_most_available_gpu()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import json
import multiprocessing
import pickle
import re
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import overload

import numpy as np
import pandas as pd
import pybnesian as pbn
from scipy.special import logsumexp, ndtr
from scipy.stats import norm

from ...bn import SemiParametricBayesianNetwork
from ...bnc import (
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
    SemiParametricNaiveBayes,
    SemiParametricTreeAugmentedNaiveBayes,
)
from ...data_handler import DATASET_NAME_LIST, DataHandler
from ...utils.constants import TRUE_CLASS_LABEL

# endregion Imports

# region Constants
# src/experiments/<group>/<module>.py -> library root
LIBRARY_ROOT_PATH = Path(__file__).resolve().parents[3]
RESULT_PATH = LIBRARY_ROOT_PATH / "data" / "results" / "inference"
# E6 (SPBNs learned from real data) was removed; the other experiments keep their numbers so that their result folders
# stay valid. E1, E5 and the rejection ablation of E4 are reported in the appendix of the article.
EXPERIMENT_NAMES = ["E1", "E2", "E3", "E4", "E5", "E7"]

# Inference methods: the arguments of ``SemiParametricBayesianNetwork.infer``, plus the posterior ``estimator``:
# "default" is the estimate of the article (mixture of exact posteriors for the Rao-Blackwellized CLG query, KDE for a sampled continuous query, frequencies for a discrete query), and "sampled_kde" draws the Rao-Blackwellized query from its exact posterior given each sample and uses a KDE (ablation E4).
METHODS = {
    "LW": {"method": "lw", "rao_blackwellize": False},
    "RBLW": {"method": "lw", "rao_blackwellize": True},
    "GS": {"method": "gibbs", "rao_blackwellize": False},
    "RBGS": {"method": "gibbs", "rao_blackwellize": True},
    # E4 ablations
    "RBLW-KDE": {"method": "lw", "rao_blackwellize": True, "estimator": "sampled_kde"},
    "RBGS-noblock": {"method": "gibbs", "rao_blackwellize": True, "block_moves": False},
    "RBGS-rej0": {
        "method": "gibbs",
        "rao_blackwellize": True,
        "max_rejection_tries": 0,
    },
    "RBGS-rej1": {
        "method": "gibbs",
        "rao_blackwellize": True,
        "max_rejection_tries": 1,
    },
    "RBGS-rej10": {
        "method": "gibbs",
        "rao_blackwellize": True,
        "max_rejection_tries": 10,
    },
    "RBGS-rej100": {
        "method": "gibbs",
        "rao_blackwellize": True,
        "max_rejection_tries": 100,
    },
}
GIBBS_DEFAULTS = {
    "n_chains": 100,
    "burn_in": 100,
    "thinning": 1,
    "max_rejection_tries": 10,
}
QUERY_TYPES = ["clg", "s_continuous", "discrete"]
# Number of points on which the posterior CDFs are evaluated to compute the Wasserstein-1 error of a continuous query, W₁ = ∫|F̂ − F*| dq
CDF_GRID_SIZE = 1024
# Batch size of the reference likelihood weighting
REFERENCE_BATCH_SIZE = 100_000
# Grid points per dimension of the grid reference, by number of unobserved continuous variables
GRID_POINTS = {1: 4001, 2: 401, 3: 101}
MIN_REFERENCE_ESS = 1000
# Samples of the reference LW whose weight is below exp(-40) times the maximum are not stored (negligible contribution)
REFERENCE_PRUNE_LOG_WEIGHT = 40.0
# A discrete task whose reference gives more than this probability to one category is near-certain (reported apart)
NEAR_CERTAIN_PROBABILITY = 0.99
MAX_REFERENCE_RHAT = 1.01
# Differences below this are numerically identical estimates, e.g., of a near-certain class,
# whose Monte Carlo standard errors are 0, so the 3-MCSE tolerance is 0 or undefined
REFERENCE_AGREEMENT_ATOL = 1e-9
# Default parallel workers of the (untimed) reference stage
DEFAULT_REFERENCE_WORKERS = 16
# Device whose time dominates each method family: LW and RB-LW evaluate the CKDE-CPDs on the GPU, while GS and RB-GS
# update the chains with mixtures on the CPU, so the timed stage runs each family on its own workers,
METHOD_DEVICE = {"lw": "gpu", "gibbs": "cpu"}
# The timed stage uses the first cores (one per GPU worker, then one per CPU worker), and the launcher runs the analyses
# on the others,
TIMED_CORES = 16
# With fewer units than CPU workers, the CPU part of a unit is split by tasks into shards, about two per worker, so the
# units of small experiments (E2, E4) do not leave workers idle,
SHARDS_PER_CPU_WORKER = 2
# endregion Constants


# region Synthetic semiparametric Bayesian networks
@dataclass(frozen=True)
class SyntheticConfig:
    """Generative configuration of a synthetic SPBN.

    The discrete variables (D0, D1, ...) are roots, followed in topological order by the CKDE variables and by the CLG variables.
    Since the CKDE variables only have discrete and CKDE parents, the sampled set S is exactly the discrete and CKDE block, and the Rao-Blackwellized set R the CLG block, so |R| / n = ``frac_clg`` up to rounding.

    Attributes:
        n_nodes: Total number of variables, discrete ones included.
        frac_clg: Fraction of CLG variables |R| / n.
        n_discrete: Number of discrete root variables.
        n_categories: Number of categories of every discrete variable.
        n_discrete_parents_R: Number of discrete variables that are parents of every CLG variable, so the number of configurations delta of the discrete parents of R is n_categories ** n_discrete_parents_R.
        max_parents: Maximum number of continuous parents of every continuous variable.
        prob_discrete_parent: Probability that the first discrete variable is a parent of each CKDE variable.
        chain_prob: Probability that each CLG variable has the previous CLG variable as parent (deep CLG chains).
        dependence: Fraction of the variance of a continuous variable explained by its continuous parents.
        n_train: Number of training instances, i.e., of kernels of the CKDE-CPDs.
        class_separation: Standard deviation of the category-specific shifts of the continuous children of discrete variables.
    """

    n_nodes: int = 20
    frac_clg: float = 0.5
    n_discrete: int = 1
    n_categories: int = 2
    n_discrete_parents_R: int = 0
    max_parents: int = 2
    prob_discrete_parent: float = 0.5
    chain_prob: float = 0.8
    dependence: float = 0.6
    n_train: int = 1000
    class_separation: float = 1.0


def _standardize(values: np.ndarray) -> np.ndarray:
    std = values.std()
    return (values - values.mean()) / (std if std > 0 else 1.0)


def _unit_noise(rng: np.random.Generator, size: int, gaussian: bool) -> np.ndarray:
    """Unit-variance noise: Gaussian for the CLG variables, Student-t (4 degrees of freedom) or a two-component mixture for the CKDE variables."""
    if gaussian:
        return rng.standard_normal(size)
    if rng.random() < 0.5:
        return rng.standard_t(4, size) / np.sqrt(2.0)
    component = rng.random(size) < 0.5
    return _standardize(
        np.where(component, -1.0, 1.0) + 0.5 * rng.standard_normal(size)
    )


def generate_synthetic_spbn(
    config: SyntheticConfig, seed: int
) -> tuple[SemiParametricBayesianNetwork, dict]:
    """Generates a synthetic SPBN: a random structure and CPD types (see ``SyntheticConfig``), training data sampled from a nonlinear generative process with that structure, and the CPDs fitted to the data with the given types.

    The ground truth of the experiments is the posterior of the fitted SPBN itself, not of the generative process.

    Returns:
        tuple[SemiParametricBayesianNetwork, dict]: The fitted SPBN and a description of the network.
    """
    rng = np.random.default_rng(seed)
    if config.n_discrete_parents_R > config.n_discrete:
        raise ValueError("n_discrete_parents_R cannot exceed n_discrete.")
    discrete_nodes = [f"D{i}" for i in range(config.n_discrete)]
    n_continuous = config.n_nodes - config.n_discrete
    n_clg = min(int(round(config.frac_clg * config.n_nodes)), n_continuous)
    ckde_nodes = [f"K{i}" for i in range(n_continuous - n_clg)]
    clg_nodes = [f"G{i}" for i in range(n_clg)]
    categories = [f"c{k}" for k in range(config.n_categories)]

    # Structure
    parents: dict[str, list[str]] = {node: [] for node in discrete_nodes}
    for i, node in enumerate(ckde_nodes):
        n_parents = rng.integers(0, min(config.max_parents, i) + 1)
        parents[node] = list(rng.choice(ckde_nodes[:i], size=n_parents, replace=False))
        if discrete_nodes and rng.random() < config.prob_discrete_parent:
            parents[node].insert(0, discrete_nodes[0])
    for i, node in enumerate(clg_nodes):
        continuous_parents = []
        if i > 0 and rng.random() < config.chain_prob:
            continuous_parents.append(clg_nodes[i - 1])
        candidates = [
            c for c in ckde_nodes + clg_nodes[:i] if c not in continuous_parents
        ]
        n_extra = rng.integers(
            1 if not continuous_parents and candidates else 0,
            max(config.max_parents - len(continuous_parents), 0) + 1,
        )
        n_extra = min(n_extra, len(candidates))
        continuous_parents += list(rng.choice(candidates, size=n_extra, replace=False))
        parents[node] = (
            discrete_nodes[: config.n_discrete_parents_R] + continuous_parents
        )

    # Data, sampled in topological order
    n = config.n_train
    data: dict[str, np.ndarray] = {}
    # The discrete parents of R follow a balanced design, so every one of their n_categories ** m configurations has about
    # n_train / n_categories ** m training instances (otherwise the CLG-CPDs of empty configurations cannot be fitted)
    m = config.n_discrete_parents_R
    if m > 0:
        configurations = rng.permutation(np.arange(n) % config.n_categories**m)
        for position, node in enumerate(discrete_nodes[:m]):
            data[node] = (
                configurations // config.n_categories**position
            ) % config.n_categories
    for node in discrete_nodes[m:]:
        probabilities = rng.dirichlet([3.0] * config.n_categories)
        data[node] = rng.choice(config.n_categories, size=n, p=probabilities)
    nonlinearities = [
        lambda s: np.sin(2.0 * s),
        lambda s: 2.0 * np.tanh(2.0 * s),
        lambda s: s**2,
        np.abs,
    ]
    for node in ckde_nodes + clg_nodes:
        is_clg = node in clg_nodes
        continuous_parents = [p for p in parents[node] if p not in discrete_nodes]
        noise = _unit_noise(rng, n, gaussian=is_clg)
        if continuous_parents:
            coefficients = rng.choice(
                [-1.0, 1.0], len(continuous_parents)
            ) * rng.uniform(0.5, 1.5, len(continuous_parents))
            signal = (
                np.column_stack([data[p] for p in continuous_parents]) @ coefficients
            )
            if not is_clg:
                signal = nonlinearities[rng.integers(len(nonlinearities))](signal)
            values = (
                np.sqrt(config.dependence) * _standardize(signal)
                + np.sqrt(1.0 - config.dependence) * noise
            )
        elif is_clg:
            values = noise
        else:
            # Root CKDE variables are bimodal
            values = _standardize(
                np.where(rng.random(n) < 0.5, -1.5, 1.5) + 0.6 * noise
            )
        for discrete_parent in [p for p in parents[node] if p in discrete_nodes]:
            shifts = rng.normal(0.0, config.class_separation, config.n_categories)
            values = values + shifts[data[discrete_parent]]
        data[node] = _standardize(values)

    df = pd.DataFrame({node: data[node] for node in ckde_nodes + clg_nodes})
    for node in discrete_nodes:
        df[node] = pd.Categorical(
            np.asarray(categories)[data[node]], categories=categories
        )

    bn = SemiParametricBayesianNetwork(
        true_label=discrete_nodes[0] if discrete_nodes else TRUE_CLASS_LABEL
    )
    for node in discrete_nodes + ckde_nodes + clg_nodes:
        bn.add_node(node)
    for node, node_parents in parents.items():
        for parent in node_parents:
            bn.add_arc(parent, node)
    for node in discrete_nodes:
        bn.set_node_type(node, pbn.DiscreteFactorType())
    for node in ckde_nodes:
        bn.set_node_type(node, pbn.CKDEType())
    for node in clg_nodes:
        bn.set_node_type(node, pbn.LinearGaussianCPDType())
    pbn.SemiparametricBN.fit(bn, df[discrete_nodes + ckde_nodes + clg_nodes])
    bn.feature_names_in_ = ckde_nodes + clg_nodes
    bn.n_features_in_ = len(bn.feature_names_in_)

    info = describe_network(bn)
    info.update({f"config_{key}": value for key, value in asdict(config).items()})
    info["network_seed"] = seed
    return bn, info


def ckde_sizes(bn: SemiParametricBayesianNetwork) -> dict[str, float]:
    """Expected number of kernels of each CKDE-CPD per evaluation: its number of training instances, or, with discrete parents, sum_c L_c**2 / sum_c L_c, the size of the configuration c of a sample drawn with the training frequencies."""
    categories = bn._categories()
    discrete = bn._discrete_nodes()
    sizes = {}
    for node in bn.nodes():
        if bn.cpd(node).type() != pbn.CKDEType():
            continue
        discrete_parents = [p for p in bn.parents(node) if p in discrete]
        counts = []
        for values in itertools.product(*(categories[p] for p in discrete_parents)):
            try:
                kernel = bn._kernel_cache(node, dict(zip(discrete_parents, values)))
            except Exception:  # configuration without training instances
                continue
            counts.append(len(kernel["offset"]))
        counts = np.asarray(counts, dtype=float)
        sizes[node] = float(np.sum(counts**2) / np.sum(counts)) if counts.sum() else 0.0
    return sizes


def describe_network(bn: SemiParametricBayesianNetwork) -> dict:
    """Sizes of the sampled and Rao-Blackwellized sets and of the CKDE-CPDs of an SPBN"""
    sampled = bn._sampled_nodes()
    rb_nodes = bn._rb_nodes(sampled)
    discrete = bn._discrete_nodes()
    categories = bn._categories() if discrete else {}
    rb_discrete_parents = {
        p for node in rb_nodes for p in bn.parents(node) if p in discrete
    }
    sizes = ckde_sizes(bn)
    node_types = [bn.cpd(node).type() for node in bn.nodes()]
    return {
        "n_nodes": len(bn.nodes()),
        "n_arcs": len(bn.arcs()),
        "n_discrete": len(discrete),
        "n_ckde": sum(t == pbn.CKDEType() for t in node_types),
        "n_clg": sum(t == pbn.LinearGaussianCPDType() for t in node_types),
        "n_sampled": len(sampled),
        "n_rb": len(rb_nodes),
        "frac_rb": len(rb_nodes) / len(bn.nodes()),
        "n_delta": (
            int(np.prod([len(categories[p]) for p in rb_discrete_parents]))
            if rb_discrete_parents
            else 1
        ),
        "mean_ckde_size": float(np.mean(list(sizes.values()))) if sizes else 0.0,
    }


def two_node_spbn(
    child_type: str, seed: int, n_train: int = 300
) -> SemiParametricBayesianNetwork:
    """x (bimodal CKDE) -> y (``child_type`` "clg" or "ckde"), the E1 check of the rejection sampling of RBGS"""
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [rng.normal(-2, 0.5, n_train // 2), rng.normal(2, 0.7, n_train - n_train // 2)]
    )
    y = 0.8 * x + np.sin(x) + 0.6 * rng.normal(size=n_train)
    bn = SemiParametricBayesianNetwork()
    bn.add_node("x")
    bn.add_node("y")
    bn.add_arc("x", "y")
    bn.set_node_type("x", pbn.CKDEType())
    bn.set_node_type(
        "y", pbn.LinearGaussianCPDType() if child_type == "clg" else pbn.CKDEType()
    )
    pbn.SemiparametricBN.fit(bn, pd.DataFrame({"x": x, "y": y}))
    bn.feature_names_in_ = ["x", "y"]
    return bn


# endregion Synthetic semiparametric Bayesian networks


# region Inference tasks
@dataclass
class Task:
    """A query variable and hard evidence, with the factors that describe the task"""

    task_id: str
    query: str
    query_type: str
    evidence: dict
    factors: dict = field(default_factory=dict)
    true_value: float | str | None = None


@dataclass(frozen=True)
class EvidenceSpec:
    """How the evidence of a task is chosen.

    Attributes:
        n_evidence: Number of evidence variables.
        location: "S" (only sampled variables), "R" (only Rao-Blackwellized variables) or "both".
        n_rb_ancestors: If set, the evidence on R is a single CLG variable with exactly this number of unobserved CLG ancestors (its excess ancestors are also observed), and the rest of the evidence is on S.
        rarity: Prior tail probability of the evidence values: 0.5 draws them jointly from the prior, and q < 0.5 sets each continuous evidence variable to its prior q or 1 - q quantile (random side).
    """

    n_evidence: int = 4
    location: str = "both"
    n_rb_ancestors: int | None = None
    rarity: float = 0.5


def node_query_type(
    bn: SemiParametricBayesianNetwork, node: str, sampled: set[str]
) -> str:
    """Query type of a node: "discrete", "clg" or "s_continuous"."""
    if node in bn._discrete_nodes():
        return "discrete"
    return "s_continuous" if node in sampled else "clg"


def unobserved_rb_ancestors(
    bn: SemiParametricBayesianNetwork, evidence: dict, sampled: set[str]
) -> int:
    """Number of unobserved CLG variables of R that are ancestors of the evidence on R, which determines the variance reduction of the RBLW weights (with 0, the RBLW and LW weights coincide)"""
    rb_nodes = set(bn._rb_nodes(sampled))
    ancestors = set()
    for node in evidence:
        if node in rb_nodes:
            ancestors |= _ancestors(bn, node) & rb_nodes
    return len(ancestors - set(evidence))


def _ancestors(bn: SemiParametricBayesianNetwork, node: str) -> set[str]:
    """All the ancestors of a node in the BN, including the node itself."""
    result, stack = set(), list(bn.parents(node))
    while stack:
        parent = stack.pop()
        if parent not in result:
            result.add(parent)
            stack.extend(bn.parents(parent))
    return result


def generate_tasks(
    bn: SemiParametricBayesianNetwork,
    n_tasks: int,
    query_types: list[str],
    spec: EvidenceSpec,
    seed: int,
    prior: pd.DataFrame,
    id_prefix: str = "",
    query_candidates: list[str] | None = None,
) -> list[Task]:
    """Generates up to ``n_tasks`` tasks per query type with the evidence of ``spec``. Infeasible combinations (e.g., no unobserved CLG variable left, or no CLG variable with enough ancestors) are skipped.

    Args:
        prior: Samples of the prior of ``bn`` (``bn.sample``), used to draw the evidence values.
        query_candidates: If given, the query is restricted to these variables.
    """
    rng = np.random.default_rng(seed)
    sampled = bn._sampled_nodes()
    rb_nodes = bn._rb_nodes(sampled)
    discrete = bn._discrete_nodes()
    pools = {
        "clg": list(rb_nodes),
        "s_continuous": [n for n in sampled if n not in discrete],
        "discrete": sorted(discrete),
    }
    topo_order = list(bn.graph().topological_sort())
    pools = {
        key: [
            n
            for n in topo_order
            if n in nodes and (query_candidates is None or n in query_candidates)
        ]
        for key, nodes in pools.items()
    }
    tasks = []
    for query_type in query_types:
        for task_index in range(n_tasks):
            task = _draw_task(
                bn, query_type, spec, rng, prior, pools, sampled, rb_nodes
            )
            if task is None:
                continue
            task.task_id = f"{id_prefix}{query_type}_{task_index}"
            tasks.append(task)
    return tasks


def _draw_task(
    bn, query_type, spec, rng, prior, pools, sampled, rb_nodes
) -> Task | None:
    """Draws a single task of the given query type and evidence spec, or returns None if the combination is infeasible."""
    if not pools[query_type]:
        return None
    query = str(rng.choice(pools[query_type]))
    s_pool = [str(n) for n in sampled if n != query]
    r_pool = [str(n) for n in rb_nodes if n != query]
    evidence_nodes: list[str] = []
    if spec.n_rb_ancestors is not None:
        rb_set = set(rb_nodes)
        counts = {n: len((_ancestors(bn, n) & rb_set) - {query}) for n in r_pool}
        candidates = [n for n, c in counts.items() if c >= spec.n_rb_ancestors]
        if not candidates:
            return None
        target = min(candidates, key=lambda n: (counts[n], rng.random()))
        ancestors = [
            n
            for n in bn.graph().topological_sort()
            if n in (_ancestors(bn, target) & rb_set) - {query}
        ]
        # Observe the earliest excess ancestors, so exactly n_rb_ancestors remain unobserved
        evidence_nodes = [target] + ancestors[: len(ancestors) - spec.n_rb_ancestors]
        n_s = max(spec.n_evidence - len(evidence_nodes), 0)
        evidence_nodes += list(
            rng.choice(s_pool, size=min(n_s, len(s_pool)), replace=False)
        )
    else:
        if spec.location == "S":
            n_s, n_r = spec.n_evidence, 0
        elif spec.location == "R":
            n_s, n_r = 0, spec.n_evidence
        else:
            n_r = spec.n_evidence // 2
            n_s = spec.n_evidence - n_r
        if n_s > len(s_pool) or n_r > len(r_pool):
            return None
        evidence_nodes = list(rng.choice(s_pool, size=n_s, replace=False)) + list(
            rng.choice(r_pool, size=n_r, replace=False)
        )
    if query in evidence_nodes or not evidence_nodes:
        return None

    row = prior.iloc[int(rng.integers(len(prior)))]
    discrete = bn._discrete_nodes()
    evidence = {}
    for node in map(str, evidence_nodes):
        if node in discrete:
            evidence[node] = str(row[node])
        elif spec.rarity >= 0.5:
            evidence[node] = float(row[node])
        else:
            q = spec.rarity if rng.random() < 0.5 else 1.0 - spec.rarity
            evidence[node] = float(np.quantile(prior[node].to_numpy(dtype=float), q))
    return Task(
        task_id="",
        query=query,
        query_type=query_type,
        evidence=evidence,
        factors={
            "n_evidence": len(evidence),
            "evidence_location": (
                spec.location if spec.n_rb_ancestors is None else "both"
            ),
            "rarity": spec.rarity,
            "n_evidence_S": sum(n in sampled for n in evidence),
            "n_evidence_R": sum(n not in sampled for n in evidence),
            "n_unobserved_rb_ancestors": unobserved_rb_ancestors(bn, evidence, sampled),
        },
    )


# endregion Inference tasks


# region Reference posteriors
@dataclass
class Reference:
    """Reference posterior of a task: moments and CDF on ``grid`` for a continuous query, probabilities for a discrete one"""

    method: str
    ok: bool
    mean: float = np.nan
    std: float = np.nan
    grid: list[float] = field(default_factory=list)
    cdf: list[float] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    mcse: float = np.nan
    effective_sample_size: float = np.nan
    message: str = ""
    n_samples: int = 0


def _evaluation_grid(mean: float, std: float, lo: float, hi: float) -> np.ndarray:
    """Grid of the CDF errors: the reference posterior range widened by 5 posterior standard deviations"""
    return np.linspace(lo - 5.0 * std, hi + 5.0 * std, CDF_GRID_SIZE)


def _reference_from_weighted_samples(
    values: np.ndarray,
    log_weights: np.ndarray,
    is_discrete: bool,
    categories: list[str],
    ess: float | None = None,
) -> Reference:
    """Reference from weighted samples; ``ess`` overrides the effective sample size of the given weights (e.g., when
    negligible samples were not stored)"""
    weights = np.exp(log_weights - logsumexp(log_weights))
    if ess is None:
        ess = float(1.0 / np.sum(weights**2))
    if is_discrete:
        probabilities = {c: float(np.sum(weights[values == c])) for c in categories}
        mcse = float(
            np.max([np.sqrt(p * (1 - p) / ess) for p in probabilities.values()])
        )
        return Reference(
            "lw",
            True,
            probabilities=probabilities,
            mcse=mcse,
            effective_sample_size=ess,
        )
    values = values.astype(float)
    mean = float(np.sum(weights * values))
    std = float(np.sqrt(np.sum(weights * (values - mean) ** 2)))
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    lo = values[order][np.searchsorted(cumulative, 1e-5)]
    hi = values[order][min(np.searchsorted(cumulative, 1 - 1e-5), len(values) - 1)]
    grid = _evaluation_grid(mean, std, lo, hi)
    index = np.searchsorted(values[order], grid, side="right")
    cdf = np.concatenate([[0.0], cumulative])[index]
    return Reference(
        "lw",
        True,
        mean=mean,
        std=std,
        grid=grid.tolist(),
        cdf=cdf.tolist(),
        mcse=std / np.sqrt(ess),
        effective_sample_size=ess,
    )


def lw_reference(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    n_samples: int,
    seed: int,
    max_samples: int | None = None,
) -> Reference:
    """Reference posterior with plain likelihood weighting (every variable sampled), in batches combined through their
    ``log_evidence``, so the unnormalized weights of all the batches are on the same scale.

    After ``n_samples``, sampling continues in batches while the effective sample size is below ``MIN_REFERENCE_ESS``, up
    to ``max_samples`` (unlikely evidence degenerates the weights, which is where the methods differ most, so a fixed budget
    would discard exactly those tasks). Samples whose weight is below exp(-``REFERENCE_PRUNE_LOG_WEIGHT``) times the
    running maximum are not stored, since their contribution is negligible, which bounds the memory of large references;
    the effective sample size is computed from all the weights.
    """
    max_samples = max(max_samples or n_samples, n_samples)
    is_discrete = task.query_type == "discrete"
    seed_sequence = np.random.SeedSequence(seed)
    values, log_weights = [], []
    log_sum, log_sum_squares, log_max = -np.inf, -np.inf, -np.inf
    drawn = 0

    def _ess() -> float:
        return (
            float(np.exp(2 * log_sum - log_sum_squares))
            if np.isfinite(log_sum)
            else 0.0
        )

    while drawn < max_samples:
        if drawn >= n_samples and _ess() >= MIN_REFERENCE_ESS:
            break
        target = n_samples if drawn < n_samples else max_samples
        batch = min(REFERENCE_BATCH_SIZE, target - drawn)
        child = seed_sequence.spawn(1)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = bn.infer(
                evidence=task.evidence,
                n_samples=batch,
                seed=int(child.generate_state(1)[0] % (2**31 - 1)),
                method="lw",
                rao_blackwellize=False,
            )
        parameters = result["parameters"]
        with np.errstate(divide="ignore"):
            batch_log_weights = (
                np.log(parameters["weights"])
                + parameters["log_evidence"]
                + np.log(batch)
            )
        column = parameters["assignments"][task.query]
        batch_values = (
            column.astype(str).to_numpy()
            if is_discrete
            else column.to_numpy(dtype=float)
        )
        log_sum = np.logaddexp(log_sum, logsumexp(batch_log_weights))
        log_sum_squares = np.logaddexp(
            log_sum_squares, logsumexp(2 * batch_log_weights)
        )
        log_max = max(log_max, float(np.max(batch_log_weights)))
        keep = batch_log_weights > log_max - REFERENCE_PRUNE_LOG_WEIGHT
        values.append(batch_values[keep])
        log_weights.append(batch_log_weights[keep])
        drawn += batch
    categories = bn._categories()[task.query] if is_discrete else []
    reference = _reference_from_weighted_samples(
        np.concatenate(values),
        np.concatenate(log_weights),
        is_discrete,
        categories,
        ess=_ess(),
    )
    reference.n_samples = drawn
    return reference


def gibbs_reference(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    n_samples: int,
    seed: int,
    n_chains: int = 200,
    burn_in: int = 1000,
    max_rhat: float = MAX_REFERENCE_RHAT,
) -> Reference:
    """Reference posterior with long chains of plain Gibbs sampling, used to check the likelihood weighting reference"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = bn.infer(
            evidence=task.evidence,
            n_samples=n_samples,
            seed=seed,
            method="gibbs",
            n_chains=n_chains,
            burn_in=burn_in,
            rao_blackwellize=False,
        )
    parameters = result["parameters"]
    column = parameters["assignments"][task.query]
    ess = parameters["effective_sample_size"].get(task.query, np.nan)
    rhat = parameters["rhat"].get(task.query, np.nan)
    # The article requires converged chains for the check (R-hat < 1.01)
    converged = bool(np.isfinite(rhat) and rhat < max_rhat)
    message = "" if converged else f"Gibbs check R-hat {rhat:.3g} >= {max_rhat}"
    if task.query_type == "discrete":
        probabilities = {
            c: float(np.mean(column.astype(str) == c))
            for c in bn._categories()[task.query]
        }
        mcse = float(
            np.max([np.sqrt(p * (1 - p) / ess) for p in probabilities.values()])
        )
        return Reference(
            "gibbs",
            converged,
            probabilities=probabilities,
            mcse=mcse,
            effective_sample_size=ess,
            message=message,
        )
    values = column.to_numpy(dtype=float)
    return Reference(
        "gibbs",
        converged,
        message=message,
        mean=float(values.mean()),
        std=float(values.std()),
        mcse=float(values.std() / np.sqrt(ess)),
        effective_sample_size=float(ess),
    )


def references_agree(difference: float, tolerance: float) -> bool:
    """Whether the reference and its check agree.

    Args:
        difference: Absolute difference of their means (continuous queries) or maximum absolute difference of their
            class probabilities (discrete queries).
        tolerance: 3 combined Monte Carlo standard errors.

    Returns:
        bool: True if the difference is within the tolerance, or below ``REFERENCE_AGREEMENT_ATOL``.
    """
    if difference <= REFERENCE_AGREEMENT_ATOL:
        return True
    return bool(np.isfinite(tolerance) and difference <= tolerance)


# Message of a check that failed because the reference and its check differ, as written by ``checked_reference``,
CHECK_DISAGREEMENT_PATTERN = re.compile(
    r"^LW and \S+ references differ by (?P<difference>\S+) > (?P<tolerance>\S+)$"
)


def repair_reference_checks(unit_paths: list[Path]) -> int:
    """Re-evaluates the failed agreement checks of finished reference stages with ``references_agree``, and rewrites the
    ``references.jsonl`` of the units whose verdict changes.

    Before ``REFERENCE_AGREEMENT_ATOL`` existed, identical near-certain estimates (Monte Carlo standard errors of 0, so a
    tolerance of 0 or NaN) were discarded. The message of a failed check records the difference and the tolerance, so
    the verdict can be corrected without recomputing the references. Other failures (low ESS, Gibbs not converged) are
    left as they are.

    Args:
        unit_paths: Unit folders; those without ``references.jsonl`` are skipped.

    Returns:
        int: Number of references whose check now passes.
    """
    repaired = 0
    for unit_path in unit_paths:
        path = unit_path / "references.jsonl"
        if not path.exists():
            continue
        rows = _read_jsonl(path)
        changed = False
        for row in rows:
            match = CHECK_DISAGREEMENT_PATTERN.match(row.get("message", ""))
            if row.get("ok") or match is None:
                continue
            if references_agree(float(match["difference"]), float(match["tolerance"])):
                row["ok"] = True
                row["message"] = f"repaired: {row['message']}"
                changed = True
                repaired += 1
        if changed:
            _write_jsonl(path, rows)
    return repaired


def checked_reference(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    n_samples: int,
    n_gibbs_samples: int,
    seed: int,
    gibbs_burn_in: int = 1000,
    max_rhat: float = MAX_REFERENCE_RHAT,
    max_samples: int | None = None,
) -> Reference:
    """Likelihood weighting reference checked against an independent reference (``max_samples``: see ``lw_reference``):
    the task is kept (``ok``) only if the likelihood weighting ESS is at least ``MIN_REFERENCE_ESS``, the check is valid and both agree within 3 combined Monte Carlo standard errors.

    The check uses long plain Gibbs chains (which must converge, R-hat < ``max_rhat``) for continuous queries, and stratified likelihood weighting (one run per category, ``n_gibbs_samples`` samples each) for discrete queries, since plain Gibbs sampling barely moves between categories that separate their children.

    With ``n_gibbs_samples <= 0`` there is no check (quick runs).
    The Rao-Blackwellized methods are never used, so the reference is independent of the methods that it evaluates.
    """
    reference = lw_reference(bn, task, n_samples, seed, max_samples)
    if reference.effective_sample_size < MIN_REFERENCE_ESS:
        reference.ok = False
        reference.message = f"LW reference ESS {reference.effective_sample_size:.0f} < {MIN_REFERENCE_ESS}"
        return reference
    if n_gibbs_samples <= 0:
        return reference
    if task.query_type == "discrete":
        check = stratified_class_reference(bn, task, n_gibbs_samples, seed + 1)
        reference.method = "lw+stratified"
    else:
        check = gibbs_reference(
            bn,
            task,
            n_gibbs_samples,
            seed + 1,
            burn_in=gibbs_burn_in,
            max_rhat=max_rhat,
        )
        reference.method = "lw+gibbs"
    if not check.ok:
        reference.ok = False
        reference.message = check.message
        return reference
    tolerance = 3.0 * np.sqrt(reference.mcse**2 + check.mcse**2)
    if task.query_type == "discrete":
        difference = max(
            abs(reference.probabilities[c] - check.probabilities[c])
            for c in reference.probabilities
        )
    else:
        difference = abs(reference.mean - check.mean)
    if not references_agree(difference, tolerance):
        reference.ok = False
        reference.message = f"LW and {check.method} references differ by {difference:.4g} > {tolerance:.4g}"
    return reference


def gaussian_reference(bn: SemiParametricBayesianNetwork, task: Task) -> Reference:
    """Exact posterior of a Gaussian Bayesian network (only CLG-CPDs), by conditioning its joint Gaussian"""
    nodes = list(bn.graph().topological_sort())
    index = {node: i for i, node in enumerate(nodes)}
    intercepts, variances = np.zeros(len(nodes)), np.zeros(len(nodes))
    coefficients = np.zeros((len(nodes), len(nodes)))
    for node in nodes:
        cpd = bn.cpd(node)
        intercepts[index[node]] = cpd.beta[0]
        variances[index[node]] = cpd.variance
        for parent, coefficient in zip(cpd.evidence(), cpd.beta[1:]):
            coefficients[index[node], index[parent]] = coefficient
    A = np.linalg.inv(np.eye(len(nodes)) - coefficients)
    mean, covariance = A @ intercepts, (A * variances) @ A.T
    e = [index[node] for node in task.evidence]
    q = index[task.query]
    values = np.array([task.evidence[node] for node in task.evidence], dtype=float)
    gain = np.linalg.solve(covariance[np.ix_(e, e)], covariance[e, q])
    posterior_mean = float(mean[q] + gain @ (values - mean[e]))
    posterior_std = float(np.sqrt(covariance[q, q] - covariance[q, e] @ gain))
    grid = _evaluation_grid(
        posterior_mean,
        posterior_std,
        posterior_mean - 5 * posterior_std,
        posterior_mean + 5 * posterior_std,
    )
    return Reference(
        "exact",
        True,
        mean=posterior_mean,
        std=posterior_std,
        grid=grid.tolist(),
        cdf=norm.cdf(grid, posterior_mean, posterior_std).tolist(),
        mcse=0.0,
        effective_sample_size=np.inf,
    )


def _joint_log_density(
    bn: SemiParametricBayesianNetwork,
    points: pd.DataFrame,
    discrete_values: dict[str, str],
) -> np.ndarray:
    """log f(x) of every row of ``points`` (all the variables), with the discrete variables fixed to ``discrete_values``"""
    discrete = bn._discrete_nodes()
    log_density = np.zeros(len(points))
    for node in bn.nodes():
        if node in discrete:
            log_density += bn._node_logl(node, points)
        else:
            kernel = bn._kernel_cache(
                node, {p: discrete_values[p] for p in bn.parents(node) if p in discrete}
            )
            log_density += bn._mixture_logl(kernel, node, points)
    return log_density


def grid_reference(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    prior: pd.DataFrame,
    chunk_size: int = 4000,
) -> Reference:
    """Reference posterior by numerical integration on a grid of the (at most 3) unobserved continuous variables, enumerating the unobserved discrete variables.
    A first pass on the prior range locates the posterior mass of each
    variable, and a second pass refines the grid on that region."""
    discrete = bn._discrete_nodes()
    categories = bn._categories() if discrete else {}
    unobserved = [n for n in bn.graph().topological_sort() if n not in task.evidence]
    continuous = [n for n in unobserved if n not in discrete]
    hidden_discrete = [n for n in unobserved if n in discrete]
    if len(continuous) > 3:
        raise ValueError(
            "The grid reference supports at most 3 unobserved continuous variables."
        )
    n_points = GRID_POINTS.get(len(continuous), 1)

    def _evaluate(axes: dict[str, np.ndarray]):
        """Unnormalized posterior on the grid: marginal of the query and of every continuous axis"""
        mesh = (
            np.meshgrid(*(axes[n] for n in continuous), indexing="ij")
            if continuous
            else []
        )
        flat = {n: m.ravel() for n, m in zip(continuous, mesh)}
        n_rows = len(next(iter(flat.values()))) if flat else 1
        configurations = list(
            itertools.product(*(categories[n] for n in hidden_discrete))
        )
        log_blocks = []
        for configuration in configurations:
            values = dict(zip(hidden_discrete, configuration))
            fixed = {
                **{n: v for n, v in task.evidence.items() if n in discrete},
                **values,
            }
            log_density = np.empty(n_rows)
            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                points = pd.DataFrame(
                    {n: flat[n][start:stop] for n in continuous}
                    | {
                        n: float(v)
                        for n, v in task.evidence.items()
                        if n not in discrete
                    },
                    index=np.arange(stop - start),
                )
                for n, v in fixed.items():
                    points[n] = pd.Categorical(
                        [v] * (stop - start), categories=categories[n]
                    )
                log_density[start:stop] = _joint_log_density(bn, points, fixed)
            log_blocks.append(log_density)
        log_blocks = np.array(log_blocks)  # (configurations, grid points)
        weights = np.exp(log_blocks - np.max(log_blocks))
        return configurations, weights.reshape(
            (len(configurations), *[len(axes[n]) for n in continuous])
        )

    def _prior_axis(node):
        values = prior[node].to_numpy(dtype=float)
        lo, hi = np.quantile(values, [0.0005, 0.9995])
        width = hi - lo
        return np.linspace(lo - 0.5 * width, hi + 0.5 * width, n_points)

    axes = {n: _prior_axis(n) for n in continuous}
    for _ in range(2):
        configurations, weights = _evaluate(axes)
        refined = {}
        for d, node in enumerate(continuous):
            marginal = weights.sum(
                axis=tuple(i for i in range(weights.ndim) if i != d + 1)
            )
            cumulative = np.cumsum(marginal) / marginal.sum()
            lo_index = max(np.searchsorted(cumulative, 1e-7) - 2, 0)
            hi_index = min(
                np.searchsorted(cumulative, 1 - 1e-7) + 2, len(axes[node]) - 1
            )
            refined[node] = np.linspace(
                axes[node][lo_index], axes[node][hi_index], n_points
            )
        axes = refined if continuous else axes
    configurations, weights = _evaluate(axes)

    if task.query_type == "discrete":
        position = hidden_discrete.index(task.query)
        totals = weights.reshape(len(configurations), -1).sum(axis=1)
        probabilities = {c: 0.0 for c in categories[task.query]}
        for configuration, total in zip(configurations, totals):
            probabilities[configuration[position]] += total
        normalizer = sum(probabilities.values())
        return Reference(
            "grid",
            True,
            probabilities={c: p / normalizer for c, p in probabilities.items()},
            mcse=0.0,
            effective_sample_size=np.inf,
        )
    d = continuous.index(task.query)
    axis = axes[task.query]
    density = weights.sum(axis=tuple(i for i in range(weights.ndim) if i != d + 1))
    density = density / np.trapezoid(density, axis)
    mean = float(np.trapezoid(axis * density, axis))
    std = float(np.sqrt(np.trapezoid((axis - mean) ** 2 * density, axis)))
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(axis))]
    )
    grid = _evaluation_grid(mean, std, axis[0], axis[-1])
    cdf = np.interp(grid, axis, cumulative, left=0.0, right=1.0)
    return Reference(
        "grid",
        True,
        mean=mean,
        std=std,
        grid=grid.tolist(),
        cdf=cdf.tolist(),
        mcse=0.0,
        effective_sample_size=np.inf,
    )


def stratified_class_reference(
    bn: SemiParametricBayesianNetwork, task: Task, n_samples: int, seed: int
) -> Reference:
    """Reference class posterior P(c | e) ∝ f(e, c) of a classifier, with one plain likelihood weighting run per class clamped as evidence, whose ``log_evidence`` estimates log f(e, c).
    Stratifying by class avoids sampling the class.

    The class is a discrete variable with only a few categories, so instead of drawing it at random, the reference fixes each category in turn and sums over them exactly.
    That removes the noise that comes from which class each sample happens to get.
    """
    categories = bn._categories()[task.query]
    log_joint, relative_variances = [], []
    for index, category in enumerate(categories):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = bn.infer(
                evidence={**task.evidence, task.query: category},
                n_samples=n_samples,
                seed=seed + index,
                method="lw",
                rao_blackwellize=False,
            )
        log_joint.append(result["parameters"]["log_evidence"])
        # Relative variance of the mean importance weight: (N / ESS - 1) / N
        ess = result["parameters"]["effective_sample_size"]
        relative_variances.append(max(1.0 / ess - 1.0 / n_samples, 0.0))
    log_joint = np.asarray(log_joint)
    probabilities = np.exp(log_joint - logsumexp(log_joint))
    # Delta method for P_c = Z_c / sum_k Z_k with independent estimates Z_k of relative variances r_k
    r = np.asarray(relative_variances)
    variances = [
        (1 - p) ** 2 * p**2 * r[c]
        + p**2 * sum(probabilities[k] ** 2 * r[k] for k in range(len(r)) if k != c)
        for c, p in enumerate(probabilities)
    ]
    return Reference(
        "stratified_lw",
        bool(np.all(np.isfinite(probabilities))),
        probabilities=dict(zip(categories, probabilities.tolist())),
        mcse=float(np.sqrt(np.max(variances))),
    )


# endregion Reference posteriors


# region Posterior estimates and errors
# Phi(-CDF_REACH) < 1e-17, so the components farther than CDF_REACH scales from a grid point contribute 0 or their full
# weight to its CDF,
CDF_REACH = 8.5


def _weighted_normal_cdf(
    grid: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray | float,
) -> np.ndarray:
    """CDF of a weighted Gaussian mixture, sum_j w_j Phi((g - m_j) / s_j), at every grid point.

    The components are sorted by mean, so each grid point only evaluates Phi for the components within ``CDF_REACH``
    times the largest scale, and adds the total weight of the components below that window. The result equals the
    direct sum up to 1e-17 per component.

    Args:
        grid: Points at which the CDF is evaluated.
        weights: Normalized weights of the components.
        means: Means of the components.
        scales: Standard deviations of the components (one for all, or one each).

    Returns:
        np.ndarray: The CDF at every grid point.
    """
    scales = np.broadcast_to(np.asarray(scales, dtype=float), means.shape)
    order = np.argsort(means, kind="stable")
    means, weights, scales = means[order], weights[order], scales[order]
    # Components with zero weight do not change the CDF,
    keep = weights > 0
    means, weights, scales = means[keep], weights[keep], scales[keep]
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    reach = CDF_REACH * (float(scales.max()) if len(scales) else 0.0)
    lows = np.searchsorted(means, grid - reach, side="left")
    highs = np.searchsorted(means, grid + reach, side="right")
    cdf = np.empty(len(grid))
    for index, (point, low, high) in enumerate(zip(grid, lows, highs)):
        window = slice(low, high)
        cdf[index] = cumulative[low] + float(
            ndtr((point - means[window]) / scales[window]) @ weights[window]
        )
    return cdf


def mixture_estimate(
    weights: np.ndarray, means: np.ndarray, stds: np.ndarray | float, grid: np.ndarray
) -> dict:
    """Moments and CDF of the Gaussian mixture sum_j w_j N(m_j, s_j**2), the estimate of a Rao-Blackwellized CLG query"""
    stds = np.broadcast_to(np.asarray(stds, dtype=float), means.shape)
    mean = float(np.sum(weights * means))
    std = float(np.sqrt(np.sum(weights * (stds**2 + (means - mean) ** 2))))
    return {
        "mean": mean,
        "std": std,
        "cdf": _weighted_normal_cdf(grid, weights, means, stds),
    }


def kde_estimate(
    weights: np.ndarray,
    samples: np.ndarray,
    grid: np.ndarray,
    effective_sample_size: float | None = None,
) -> dict:
    """Moments and CDF of the weighted Gaussian KDE of the article, with the bandwidth of ``SemiParametricBayesianNetwork._kde_bandwidth``; the variance of the KDE is the weighted variance plus h**2"""
    bandwidth = SemiParametricBayesianNetwork._kde_bandwidth(
        samples, weights, effective_sample_size
    )
    mean = float(np.sum(weights * samples))
    variance = float(np.sum(weights * (samples - mean) ** 2))
    return {
        "mean": mean,
        "std": float(np.sqrt(variance + bandwidth**2)),
        "cdf": _weighted_normal_cdf(grid, weights, samples, bandwidth),
        "bandwidth": bandwidth,
    }


def estimate_posterior(
    bn: SemiParametricBayesianNetwork,
    infer_dict: dict,
    task: Task,
    reference: Reference,
    estimator: str = "default",
    seed: int = 0,
) -> dict:
    """Posterior estimate of the query (Equation posterior_estimate of the article) and its errors with respect to the reference:
    the mean and standard deviation errors and the Wasserstein-1 distance normalized by the reference standard deviation for continuous queries, and the maximum absolute probability error for discrete queries.
    """
    parameters = infer_dict["parameters"]
    weights = np.asarray(parameters["weights"], dtype=float)
    column = parameters["assignments"][task.query]
    if task.query_type == "discrete":
        values = column.astype(str).to_numpy()
        probabilities = {
            c: float(np.sum(weights[values == c])) for c in bn._categories()[task.query]
        }
        row = {"probabilities": json.dumps(probabilities)}
        if reference.ok and reference.probabilities:
            row["prob_error"] = max(
                abs(probabilities[c] - reference.probabilities[c])
                for c in probabilities
            )
        return row

    grid = np.asarray(reference.grid, dtype=float)
    values = column.to_numpy(dtype=float)
    posterior_std = parameters["posterior_std"]
    ess = None
    if infer_dict["method"] == "gibbs":
        ess = parameters["effective_sample_size"].get(task.query)
    if task.query in posterior_std and estimator == "sampled_kde":
        # Ablation: draw the query from its exact posterior given each sample, then a KDE as if it had been sampled
        rng = np.random.default_rng(seed)
        draws = values + np.asarray(posterior_std[task.query]) * rng.standard_normal(
            len(values)
        )
        estimate = kde_estimate(weights, draws, grid, ess)
    elif task.query in posterior_std:
        estimate = mixture_estimate(weights, values, posterior_std[task.query], grid)
    else:
        estimate = kde_estimate(weights, values, grid, ess)
    row = {
        "mean": estimate["mean"],
        "std": estimate["std"],
        "bandwidth": estimate.get("bandwidth", np.nan),
    }
    if reference.ok and len(grid) > 0:
        row["err_mean"] = (estimate["mean"] - reference.mean) / reference.std
        row["err_std"] = (estimate["std"] - reference.std) / reference.std
        row["w1"] = (
            float(
                np.trapezoid(np.abs(estimate["cdf"] - np.asarray(reference.cdf)), grid)
            )
            / reference.std
        )
    return row


# endregion Posterior estimates and errors


# region Inference runs
@contextmanager
def count_kernel_evaluations():
    """Counts the kernel evaluations of the CKDE-CPDs computed with the cached mixtures (``_conditional_mixture``, used by the Gibbs samplers):
    the number of rows times the number of kernels of every call.
    Yields a one-element list.
    """
    original = SemiParametricBayesianNetwork.__dict__["_conditional_mixture"]
    counter = [0]

    def _counted(kernel, state):
        if kernel["type"] == "ckde":
            counter[0] += len(state) * len(kernel["offset"])
        return original.__func__(kernel, state)

    SemiParametricBayesianNetwork._conditional_mixture = staticmethod(_counted)
    try:
        yield counter
    finally:
        SemiParametricBayesianNetwork._conditional_mixture = original


@overload
def run_method(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    method_name: str,
    n_samples: int,
    seed: int,
    reference: Reference,
    sizes: dict[str, float],
    gibbs_options: dict,
) -> dict: ...


@overload
def run_method(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    method_name: str,
    n_samples: list[int],
    seed: int,
    reference: Reference,
    sizes: dict[str, float],
    gibbs_options: dict,
) -> list[dict]: ...


def run_method(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    method_name: str,
    n_samples: int | list[int],
    seed: int,
    reference: Reference,
    sizes: dict[str, float],
    gibbs_options: dict,
) -> dict | list[dict]:
    """Runs one inference method on one task and returns the result row: runtime, kernel evaluations, posterior
    estimate, errors with respect to the reference and diagnostics.

    The repetition seed is shared by all the methods, so likelihood weighting and RBLW draw the same values of the
    sampled variables (common random numbers) and the comparisons are paired.

    Args:
        bn: The network.
        task: The task.
        method_name: A key of ``METHODS``.
        n_samples: Number of samples, or a list of them. With a list, a Gibbs method runs its chains once up to the
            largest and records the results and the runtime at the others (``checkpoints`` of ``infer``), which are the
            same as separate runs with the same seed; likelihood weighting runs once per number of samples.
        seed: The repetition seed.
        reference: The reference posterior of the task.
        sizes: Kernels per CKDE-CPD (``ckde_sizes``).
        gibbs_options: Options of the Gibbs samplers.

    Returns:
        dict | list[dict]: The result row, or one row per number of samples if ``n_samples`` is a list.
    """
    if isinstance(n_samples, list):
        if METHODS[method_name]["method"] != "gibbs":
            return [
                _run_once(
                    bn, task, method_name, n, seed, reference, sizes, gibbs_options
                )
                for n in n_samples
            ]
        return _run_gibbs_checkpoints(
            bn, task, method_name, n_samples, seed, reference, sizes, gibbs_options
        )
    return _run_once(
        bn, task, method_name, n_samples, seed, reference, sizes, gibbs_options
    )


def _run_once(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    method_name: str,
    n_samples: int,
    seed: int,
    reference: Reference,
    sizes: dict[str, float],
    gibbs_options: dict,
) -> dict:
    """One run of a method with ``n_samples`` samples (see ``run_method``)."""
    spec = dict(METHODS[method_name])
    kwargs = {"evidence": task.evidence, "n_samples": n_samples, "seed": seed}
    if spec["method"] == "gibbs":
        kwargs = {**gibbs_options, **kwargs}
    spec_kwargs = {k: v for k, v in spec.items() if k != "estimator"}
    with warnings.catch_warnings(), count_kernel_evaluations() as counter:
        warnings.simplefilter("ignore", RuntimeWarning)
        start = time.perf_counter()
        # The options of the method (e.g., max_rejection_tries) take precedence over the Gibbs defaults,
        infer_dict = bn.infer(**{**kwargs, **spec_kwargs})
        runtime = time.perf_counter() - start
    return _result_row(
        bn,
        task,
        method_name,
        n_samples,
        seed,
        reference,
        sizes,
        infer_dict,
        runtime,
        counter[0],
        kwargs.get("n_chains", 0),
    )


def _run_gibbs_checkpoints(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    method_name: str,
    sample_sizes: list[int],
    seed: int,
    reference: Reference,
    sizes: dict[str, float],
    gibbs_options: dict,
) -> list[dict]:
    """Runs a Gibbs method once up to the largest number of samples and returns one result row per number of samples,
    from the checkpoints of the chains (see ``run_method``)."""
    spec = {k: v for k, v in METHODS[method_name].items() if k != "estimator"}
    kwargs = {
        **gibbs_options,
        "evidence": task.evidence,
        "n_samples": max(sample_sizes),
        "seed": seed,
    }
    with warnings.catch_warnings(), count_kernel_evaluations() as counter:
        warnings.simplefilter("ignore", RuntimeWarning)
        infer_dict = bn.infer(
            **{**kwargs, **spec},
            checkpoints=sample_sizes,
            checkpoint_callback=lambda _: {"kernel_evaluations": counter[0]},
        )
    rows = []
    for n_samples in sample_sizes:
        checkpoint = infer_dict["checkpoints"][n_samples]
        rows.append(
            _result_row(
                bn,
                task,
                method_name,
                n_samples,
                seed,
                reference,
                sizes,
                {**infer_dict, "parameters": checkpoint["parameters"]},
                checkpoint["runtime"],
                checkpoint["callback"]["kernel_evaluations"],
                kwargs["n_chains"],
            )
        )
    return rows


def _result_row(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    method_name: str,
    n_samples: int,
    seed: int,
    reference: Reference,
    sizes: dict[str, float],
    infer_dict: dict,
    runtime: float,
    counted_kernel_evaluations: int,
    n_chains: int,
) -> dict:
    """Result row of one run (see ``run_method``) from its inference results and cost."""
    spec = METHODS[method_name]
    estimator = spec.get("estimator", "default")
    parameters = infer_dict["parameters"]
    sampled = bn._sampled_nodes(spec["rao_blackwellize"])
    ckde_cost = sum(size for node, size in sizes.items() if node in sampled)
    if spec["method"] == "lw":
        # Every sample evaluates or samples each CKDE-CPD once (all of them are in S)
        kernel_evaluations = n_samples * ckde_cost
    else:
        # The chains are initialized by forward sampling with the CKDE-CPDs of pybnesian
        kernel_evaluations = counted_kernel_evaluations + n_chains * sum(
            size
            for node, size in sizes.items()
            if node in sampled and node not in task.evidence
        )
    row = {
        "method": method_name,
        "n_samples": n_samples,
        "seed": seed,
        "runtime": runtime,
        "kernel_evaluations": kernel_evaluations,
    }
    row.update(estimate_posterior(bn, infer_dict, task, reference, estimator, seed))
    if spec["method"] == "lw":
        row["n_eff"] = parameters["effective_sample_size"]
        row["log_evidence"] = parameters["log_evidence"]
    else:
        row["rhat"] = parameters["rhat"].get(task.query, np.nan)
        row["n_eff"] = parameters["effective_sample_size"].get(task.query, np.nan)
        row["tail_ess"] = parameters["tail_effective_sample_size"].get(
            task.query, np.nan
        )
        finite_rhat = [r for r in parameters["rhat"].values() if np.isfinite(r)]
        row["max_rhat"] = max(finite_rhat) if finite_rhat else np.nan
        for rate in ["exact_rate", "acceptance_rate", "block_acceptance_rate"]:
            values = list(parameters[rate].values())
            row[f"mean_{rate}"] = float(np.mean(values)) if values else np.nan
    return row


# endregion Inference runs


# region Experiment units
@dataclass
class UnitSpec:
    """An independent unit of an experiment: a synthetic network, or a dataset and classifier and its tasks and runs.

    Attributes:
        unit_id: Unique identifier within the experiment (also the folder name).
        kind: "synthetic", "gaussian", "two_node" or "classifier".
        factors: Experimental factors of the unit (e.g., {"factor": "frac_clg", "level": 0.3}), copied to every row.
        config: Synthetic configuration (``SyntheticConfig`` as a dict), or the dataset and model.
        seed: Seed of the network and of the tasks.
        evidence: ``EvidenceSpec`` as a dict.
        query_types: Query types of the tasks.
        n_tasks: Tasks per query type.
        methods: Method names of ``METHODS``.
        sample_sizes: Sample sizes per method family ("lw", "gibbs").
        n_reps: Repetitions per method and sample size.
        reference: "exact", "grid", "lw+gibbs", "stratified_lw" or "none".
    """

    unit_id: str
    kind: str
    factors: dict
    config: dict
    seed: int
    evidence: dict
    query_types: list[str]
    n_tasks: int
    methods: list[str]
    sample_sizes: dict[str, list[int]]
    n_reps: int
    reference: str
    n_reference: int = 10_000_000
    # The LW reference continues up to this many samples while its ESS is below MIN_REFERENCE_ESS
    n_reference_max: int = 100_000_000
    n_reference_gibbs: int = 200_000
    reference_burn_in: int = 1000
    reference_max_rhat: float = MAX_REFERENCE_RHAT
    n_prior: int = 20_000
    gibbs_options: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def build_network(unit: UnitSpec) -> tuple[SemiParametricBayesianNetwork, dict]:
    if unit.kind in ("synthetic", "gaussian"):
        return generate_synthetic_spbn(SyntheticConfig(**unit.config), unit.seed)
    if unit.kind == "two_node":
        bn = two_node_spbn(unit.config["child_type"], unit.seed)
        return bn, describe_network(bn)
    raise ValueError(f"Unknown unit kind {unit.kind!r}.")


def compute_reference(bn, task: Task, unit: UnitSpec, prior: pd.DataFrame) -> Reference:
    try:
        match unit.reference:
            case "exact":
                return gaussian_reference(bn, task)
            case "grid":
                return grid_reference(bn, task, prior)
            case "lw+gibbs":
                return checked_reference(
                    bn,
                    task,
                    unit.n_reference,
                    unit.n_reference_gibbs,
                    unit.seed + 7919,
                    unit.reference_burn_in,
                    unit.reference_max_rhat,
                    unit.n_reference_max,
                )
            case "lw":
                return checked_reference(
                    bn,
                    task,
                    unit.n_reference,
                    0,
                    unit.seed + 7919,
                    max_samples=unit.n_reference_max,
                )
            case "none":
                return Reference("none", False)
    except (
        Exception
    ) as error:  # e.g., singular covariance or numerical failure: the task is excluded
        return Reference(
            unit.reference, False, message=f"{type(error).__name__}: {error}"
        )
    raise ValueError(f"Unknown reference {unit.reference!r}.")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _save_network(bn, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(bn, f, pickle.HIGHEST_PROTOCOL)


def _load_network(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _timing_device() -> dict:
    """GPU (name and identifier) and CPU cores of the process that times the runs"""
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    return {
        "device": opencl_device_name(),
        "device_id": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cpu_cores": ",".join(map(str, cores)),
    }


def _log_progress(unit_id: str, done: int, total: int, start: float) -> None:
    """Prints the progress of the reference stage of a unit, so a slow unit can be told apart from a frozen one.

    Args:
        unit_id: The unit.
        done: Tasks whose reference is finished.
        total: Tasks of the unit.
        start: ``time.perf_counter()`` at the start of the unit.
    """
    elapsed = time.perf_counter() - start
    remaining = elapsed / done * (total - done) if done else float("nan")
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} {unit_id}: reference {done}/{total} "
        f"({elapsed / 60:.1f} min, about {remaining / 60:.0f} min left)",
        flush=True,
    )


def prepare_unit(unit: UnitSpec, unit_path: Path) -> int:
    """Reference stage of a unit, which is not timed and can run fully in parallel: builds the network (generated, learned
    or fitted), draws the tasks and computes their references, and writes ``network.pkl``, ``network.json``,
    ``tasks.jsonl`` and ``references.jsonl``. The ``references_done`` marker is written last, so an interrupted stage is
    run again. Returns the number of tasks (-1 if the stage was already done)."""
    # A finished unit keeps the references its results were computed with (units run before the stages were split have
    # no references_done marker)
    if (unit_path / "references_done").exists() or (unit_path / "done").exists():
        return -1
    unit_path.mkdir(parents=True, exist_ok=True)
    if unit.kind == "classifier":
        n_tasks = prepare_classifier_unit(unit, unit_path)
        (unit_path / "references_done").touch()
        return n_tasks

    bn, info = build_network(unit)
    info.update({"unit_id": unit.unit_id, "kind": unit.kind, **unit.factors})
    prior = bn.sample(unit.n_prior, seed=unit.seed)
    tasks = generate_tasks(
        bn,
        unit.n_tasks,
        unit.query_types,
        EvidenceSpec(**unit.evidence),
        unit.seed + 1,
        prior,
        id_prefix=f"{unit.unit_id}/",
        query_candidates=unit.extra.get("query_candidates"),
    )
    start = time.perf_counter()
    references = []
    for index, task in enumerate(tasks):
        references.append(compute_reference(bn, task, unit, prior))
        _log_progress(unit.unit_id, index + 1, len(tasks), start)

    _save_network(bn, unit_path / "network.pkl")
    with open(unit_path / "network.json", "w") as f:
        json.dump(info, f, default=_json_default, indent=2)
    _write_jsonl(unit_path / "tasks.jsonl", [asdict(task) for task in tasks])
    _write_jsonl(
        unit_path / "references.jsonl",
        [{"task_id": t.task_id, **asdict(r)} for t, r in zip(tasks, references)],
    )
    (unit_path / "references_done").touch()
    return len(tasks)


def _unit_devices(unit: UnitSpec) -> list[str]:
    """Devices of the timed stage of a unit (see ``METHOD_DEVICE``).

    Args:
        unit: The unit.

    Returns:
        list[str]: "cpu" and/or "gpu"; the imputation baselines of E7 run on the CPU.
    """
    devices = {METHOD_DEVICE[METHODS[m]["method"]] for m in unit.methods}
    if unit.kind == "classifier":
        devices.add("cpu")
    return sorted(devices)


def _runs_on(method_name: str, device: str | None) -> bool:
    """Whether a method belongs to the part of the timed stage of ``device`` (None: every method)."""
    return device is None or METHOD_DEVICE[METHODS[method_name]["method"]] == device


def _write_atomically(path: Path, text: str) -> None:
    """Writes a file through a temporary one, so two processes that write the same content never leave it truncated."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _read_results(paths: list[Path]) -> pd.DataFrame:
    """Concatenates result files, skipping the empty ones (units or shards without valid tasks)."""
    frames = [pd.read_csv(p) for p in paths if p.stat().st_size > 1]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _finish_runs(
    unit: UnitSpec,
    unit_path: Path,
    rows: list[dict],
    device: str | None,
    shard: tuple[int, int] | None = None,
) -> None:
    """Writes the results of the timed stage of a unit, or of its part on ``device``.

    With a device, its rows go to ``results_<device>.csv``, its timing device to ``timing_<device>.json`` and its marker
    ``done_<device>`` is written; once every device of the unit is done, their results are concatenated in
    ``results.csv`` and the ``done`` marker is written.

    Args:
        unit: The unit.
        unit_path: Its folder.
        rows: The result rows of the methods that ran.
        device: "cpu", "gpu" or None (every method in one process).
        shard: (index, count) of the shard of the tasks that ran, whose rows go to ``results_<device>_<index>of<count>.csv``
            with the marker ``done_<device>_<index>of<count>``; once every shard is done, they are concatenated in
            ``results_<device>.csv``. None if every task ran.
    """
    if device is None:
        pd.DataFrame(rows).to_csv(unit_path / "results.csv", index=False)
        info = json.loads((unit_path / "network.json").read_text())
        # The device that times the runs, which may differ from the one of the reference stage
        info.update(_timing_device())
        _write_atomically(
            unit_path / "network.json",
            json.dumps(info, default=_json_default, indent=2),
        )
        (unit_path / "done").touch()
        return
    if shard is not None and shard[1] > 1:
        index, count = shard
        pd.DataFrame(rows).to_csv(
            unit_path / f"results_{device}_{index}of{count}.csv", index=False
        )
        (unit_path / f"done_{device}_{index}of{count}").touch()
        tags = [f"{device}_{i}of{count}" for i in range(count)]
        if not all((unit_path / f"done_{tag}").exists() for tag in tags):
            return
        part = _read_results([unit_path / f"results_{tag}.csv" for tag in tags])
        _write_atomically(unit_path / f"results_{device}.csv", part.to_csv(index=False))
    else:
        pd.DataFrame(rows).to_csv(unit_path / f"results_{device}.csv", index=False)
    (unit_path / f"timing_{device}.json").write_text(json.dumps(_timing_device()))
    (unit_path / f"done_{device}").touch()
    devices = _unit_devices(unit)
    if not all((unit_path / f"done_{d}").exists() for d in devices):
        return
    results = _read_results([unit_path / f"results_{d}.csv" for d in devices])
    _write_atomically(unit_path / "results.csv", results.to_csv(index=False))
    info = json.loads((unit_path / "network.json").read_text())
    for d in devices:
        timing = json.loads((unit_path / f"timing_{d}.json").read_text())
        info.update({f"{d}_runs_{key}": value for key, value in timing.items()})
    # The GPU of the LW family is the timing device of networks.csv, as before the split,
    info.update(
        json.loads(
            (
                unit_path / f"timing_{'gpu' if 'gpu' in devices else 'cpu'}.json"
            ).read_text()
        )
    )
    _write_atomically(
        unit_path / "network.json", json.dumps(info, default=_json_default, indent=2)
    )
    (unit_path / "done").touch()


def run_unit(
    unit: UnitSpec,
    unit_path: Path,
    device: str | None = None,
    shard: tuple[int, int] | None = None,
) -> int:
    """Timed stage of a unit: runs the inference methods on the tasks of ``prepare_unit`` (computing them first if
    needed) and writes ``results.csv``. The ``done`` marker is written last, so an interrupted stage is run again.

    Args:
        unit: The unit.
        unit_path: Its folder.
        device: Run only the methods of this device ("gpu": LW family, "cpu": Gibbs family and imputation), so that both
            parts run on their own workers at the same time (see ``_finish_runs``). None runs every method.
        shard: (index, count): run only the tasks whose position modulo ``count`` is ``index``, so the part of a unit on
            ``device`` can be split over several workers. None runs every task.

    Returns:
        int: Number of result rows (-1 if the unit, or its part on ``device``, or the shard, was already done or has
        nothing to run).
    """
    if (unit_path / "done").exists():
        return -1
    if device is not None and (
        device not in _unit_devices(unit) or (unit_path / f"done_{device}").exists()
    ):
        return -1
    if (
        device is not None
        and shard is not None
        and (unit_path / f"done_{device}_{shard[0]}of{shard[1]}").exists()
    ):
        return -1
    prepare_unit(unit, unit_path)
    if unit.kind == "classifier":
        rows = run_classifier_unit(unit, unit_path, device, shard)
        _finish_runs(unit, unit_path, rows, device, shard)
        return len(rows)

    bn = _load_network(unit_path / "network.pkl")
    info = json.loads((unit_path / "network.json").read_text())
    tasks = [Task(**row) for row in _read_jsonl(unit_path / "tasks.jsonl")]
    references = {
        row.pop("task_id"): Reference(**row)
        for row in _read_jsonl(unit_path / "references.jsonl")
    }
    sizes = ckde_sizes(bn)
    gibbs_options = {**GIBBS_DEFAULTS, **unit.gibbs_options}

    result_rows = []
    for position, task in enumerate(tasks):
        if shard is not None and position % shard[1] != shard[0]:
            continue
        reference = references[task.task_id]
        if unit.reference != "none" and not reference.ok:
            continue
        base = {
            "unit_id": unit.unit_id,
            "task_id": task.task_id,
            "query": task.query,
            "query_type": task.query_type,
            "reference_method": reference.method,
            # Near-certain discrete tasks (above NEAR_CERTAIN_PROBABILITY) are reported apart by the analysis
            "reference_max_probability": (
                max(reference.probabilities.values())
                if reference.probabilities
                else np.nan
            ),
            **unit.factors,
            **task.factors,
            **{
                key: info[key]
                for key in ["n_nodes", "n_rb", "frac_rb", "n_delta", "mean_ckde_size"]
            },
        }
        for method_name in unit.methods:
            if not _runs_on(method_name, device):
                continue
            family = METHODS[method_name]["method"]
            for rep in range(unit.n_reps):
                # One call per repetition: a Gibbs method runs its chains once for all the sample sizes
                rows = run_method(
                    bn,
                    task,
                    method_name,
                    list(unit.sample_sizes[family]),
                    rep,
                    reference,
                    sizes,
                    gibbs_options,
                )
                result_rows.extend({**base, **row} for row in rows)

    _finish_runs(unit, unit_path, result_rows, device, shard)
    return len(result_rows)


def run_unit_gpu(
    unit: UnitSpec, unit_path: Path, shard: tuple[int, int] | None = None
) -> int:
    """GPU part of the timed stage (LW family), see ``run_unit``."""
    return run_unit(unit, unit_path, "gpu", shard)


def run_unit_cpu(
    unit: UnitSpec, unit_path: Path, shard: tuple[int, int] | None = None
) -> int:
    """CPU part of the timed stage (Gibbs family and imputation), see ``run_unit``."""
    return run_unit(unit, unit_path, "cpu", shard)


# endregion Experiment units


# region Real datasets (E7)
def load_dataset(
    dataset_name: str, seed: int, max_train_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test partitions of a public dataset (continuous features and ``TRUE_CLASS_LABEL``)"""
    handler = DataHandler.from_public_data(
        dataset_name, seed=seed, max_train_data_size=max_train_size
    )
    train = handler.train_attack_data.reset_index(drop=True)
    test = handler.test_data_dict["test_full"].reset_index(drop=True)
    features = [c for c in train.columns if c != TRUE_CLASS_LABEL]
    train[features] = train[features].astype(float)
    test[features] = test[features].astype(float)
    return train, test


CLASSIFIERS = {
    "SP-NB": SemiParametricNaiveBayes,
    "SP-TAN": SemiParametricTreeAugmentedNaiveBayes,
    "SP-BAN": SemiParametricBayesianNetworkAugmentedNaiveBayes,
}


def prepare_classifier_unit(unit: UnitSpec, unit_path: Path) -> int:
    """Reference stage of E7: fits the classifier, removes features of the test instances completely at random, and
    computes the reference class posteriors with stratified likelihood weighting. Besides the files of ``prepare_unit``,
    it writes ``train_features.csv`` (for the imputation baselines) and ``test_rows.csv`` (the selected test instances).
    Returns the number of tasks."""
    dataset_name, model_name = unit.config["dataset"], unit.config["model"]
    train, test = load_dataset(
        dataset_name, unit.seed, unit.config.get("max_train_size", 2000)
    )
    features = [c for c in train.columns if c != TRUE_CLASS_LABEL]
    model = CLASSIFIERS[model_name](seed=unit.seed)
    model.fit(train[features], train[TRUE_CLASS_LABEL])
    model_features = list(model.feature_names_in_)
    # Zero-variance features are not variables of the network, so they cannot be evidence
    network_features = [f for f in model_features if f in model.nodes()]
    info = describe_network(model)
    info.update(
        {
            "unit_id": unit.unit_id,
            "dataset": dataset_name,
            "model": model_name,
            **unit.factors,
        }
    )

    rng = np.random.default_rng(unit.seed + 1)
    test_indices = rng.choice(
        len(test), size=min(unit.n_tasks, len(test)), replace=False
    )
    test_rows = test.iloc[test_indices].reset_index(drop=True)
    tasks, references = [], []
    n_tasks = len(unit.extra["missing_rates"]) * len(test_rows)
    start = time.perf_counter()
    for missing_rate in unit.extra["missing_rates"]:
        for task_index in range(len(test_rows)):
            row = test_rows.iloc[task_index]
            missing = rng.random(len(network_features)) < missing_rate
            if missing.all():
                missing[rng.integers(len(missing))] = False
            if not missing.any():
                missing[rng.integers(len(missing))] = True
            evidence = {
                f: float(row[f]) for f, m in zip(network_features, missing) if not m
            }
            task = Task(
                task_id=f"{unit.unit_id}/{missing_rate}/{task_index}",
                query=model.true_label,
                query_type="discrete",
                evidence=evidence,
                factors={
                    "missing_rate": missing_rate,
                    "n_missing": int(missing.sum()),
                    "test_row": task_index,
                },
                true_value=str(row[TRUE_CLASS_LABEL]),
            )
            tasks.append(task)
            references.append(
                stratified_class_reference(
                    model, task, unit.n_reference, unit.seed + 7919
                )
            )
            # Every 10% of the tasks, since E7 has many cheap tasks per unit
            if len(tasks) % max(n_tasks // 10, 1) == 0 or len(tasks) == n_tasks:
                _log_progress(unit.unit_id, len(tasks), n_tasks, start)

    _save_network(model, unit_path / "network.pkl")
    train[model_features].to_csv(unit_path / "train_features.csv", index=False)
    test_rows.to_csv(unit_path / "test_rows.csv", index=False)
    with open(unit_path / "network.json", "w") as f:
        json.dump(info, f, default=_json_default, indent=2)
    _write_jsonl(unit_path / "tasks.jsonl", [asdict(task) for task in tasks])
    _write_jsonl(
        unit_path / "references.jsonl",
        [{"task_id": t.task_id, **asdict(r)} for t, r in zip(tasks, references)],
    )
    return len(tasks)


def run_classifier_unit(
    unit: UnitSpec,
    unit_path: Path,
    device: str | None = None,
    shard: tuple[int, int] | None = None,
) -> list[dict]:
    """Timed stage of E7: class posterior of an SPBN classifier with features missing completely at random, estimated
    with the inference methods, and with mean and k-nearest-neighbors imputation followed by the exact class posterior
    of the classifier (on the tasks and references of ``prepare_classifier_unit``).

    E7 makes two comparisons on the task that motivates SPBN classifiers, predicting the class when some features are
    missing:
    - RB vs non-RB inference (the paper's claim), through the error of the class posterior with respect to the reference.
    - Inference vs imputation (whether inference is needed at all), through the log-loss, Brier score and accuracy with
      respect to the true labels.

    Args:
        unit: The unit.
        unit_path: Its folder.
        device: Run only the methods of this device (the imputation baselines run on the CPU), see ``run_unit``.
        shard: Run only the tasks of this shard, see ``run_unit``.

    Returns:
        list[dict]: The result rows.
    """
    from sklearn.impute import KNNImputer

    dataset_name, model_name = unit.config["dataset"], unit.config["model"]
    model = _load_network(unit_path / "network.pkl")
    model_features = list(model.feature_names_in_)
    network_features = [f for f in model_features if f in model.nodes()]
    classes = [str(c) for c in model.classes_]
    train_features = pd.read_csv(unit_path / "train_features.csv")
    test_rows = pd.read_csv(unit_path / "test_rows.csv")
    tasks = [Task(**row) for row in _read_jsonl(unit_path / "tasks.jsonl")]
    references = {
        row.pop("task_id"): Reference(**row)
        for row in _read_jsonl(unit_path / "references.jsonl")
    }
    sizes = ckde_sizes(model)
    gibbs_options = {**GIBBS_DEFAULTS, **unit.gibbs_options}
    imputers = {
        "MeanImpute": train_features.mean(),
        "kNNImpute": KNNImputer(n_neighbors=5).fit(train_features),
    }

    rows = []
    for position, task in enumerate(tasks):
        if shard is not None and position % shard[1] != shard[0]:
            continue
        reference = references[task.task_id]
        row = test_rows.iloc[task.factors["test_row"]]
        missing = np.array([f not in task.evidence for f in network_features])
        base = {
            "unit_id": unit.unit_id,
            "task_id": task.task_id,
            "dataset": dataset_name,
            "model": model_name,
            "true_label": task.true_value,
            **unit.factors,
            **task.factors,
        }

        def _scores(probabilities: dict[str, float]) -> dict:
            p = np.array([probabilities.get(c, 0.0) for c in classes])
            truth = np.array([c == task.true_value for c in classes], dtype=float)
            scores = {
                "log_loss": float(-np.log(max(p[truth == 1].sum(), 1e-12))),
                "brier": float(np.sum((p - truth) ** 2)),
                "correct": float(classes[int(np.argmax(p))] == task.true_value),
                "probabilities": json.dumps(dict(zip(classes, p.tolist()))),
            }
            if reference.ok:
                scores["prob_error"] = float(
                    max(
                        abs(probabilities.get(c, 0.0) - reference.probabilities[c])
                        for c in classes
                    )
                )
            return scores

        for method_name in unit.methods:
            if not _runs_on(method_name, device):
                continue
            family = METHODS[method_name]["method"]
            for rep in range(unit.n_reps):
                for result in run_method(
                    model,
                    task,
                    method_name,
                    list(unit.sample_sizes[family]),
                    rep,
                    reference,
                    sizes,
                    gibbs_options,
                ):
                    probabilities = json.loads(result.pop("probabilities"))
                    result.pop("prob_error", None)
                    rows.append({**base, **result, **_scores(probabilities)})
        for imputer_name, imputer in imputers.items():
            if device == "gpu":
                break
            x = pd.DataFrame(
                [row[model_features].to_numpy(dtype=float)], columns=model_features
            )
            x.loc[0, np.asarray(network_features)[missing]] = np.nan
            start = time.perf_counter()
            if imputer_name == "MeanImpute":
                x = x.fillna(imputer)
            else:
                x = pd.DataFrame(imputer.transform(x), columns=model_features)
            probabilities = model.predict_proba(x)[0]
            runtime = time.perf_counter() - start
            rows.append(
                {
                    **base,
                    "method": imputer_name,
                    "n_samples": 0,
                    "seed": 0,
                    "runtime": runtime,
                    **_scores(dict(zip(classes, probabilities.tolist()))),
                }
            )

    return rows


# endregion Real datasets (E7)


# region Experiment definitions
# Levels of the network factors, varied one at a time from the base configuration. E3 (Table tab:regime_factors) varies
# ``REGIME_NETWORK_FACTORS``, the factors with a predicted effect on the gain; E5 (appendix) varies ``SCALING_FACTORS``.
NETWORK_FACTORS = {
    "frac_clg": [0.1, 0.3, 0.5, 0.7, 0.9],
    "dependence": [0.3, 0.6, 0.9],
    "n_train": [200, 1000, 5000],
    "n_discrete_parents_R": [0, 1, 2, 3],
    "n_nodes": [10, 20, 50, 100],
}
REGIME_NETWORK_FACTORS = ["frac_clg", "n_discrete_parents_R"]
SCALING_FACTORS = ["n_nodes", "frac_clg", "n_train", "n_discrete_parents_R"]
# Levels of the regime factors that are the base configuration and base evidence (n_rb_ancestors has no base level,
# since it uses 40-variable networks); keep in sync with analysis.REGIME_BASE_LEVELS
REGIME_BASE_LEVELS = {
    "frac_clg": 0.5,
    "n_discrete_parents_R": 0,
    "location": "both",
    "rarity": 0.5,
}
EVIDENCE_FACTORS = {
    "location": ["S", "both", "R"],
    "n_rb_ancestors": [0, 2, 5, 10],
    "rarity": [0.5, 0.1, 0.01, 0.001],
}


def _network_factor_config(factor: str, level) -> dict:
    """Configuration of a synthetic network with one factor varied from the base configuration (see ``NETWORK_FACTORS``)"""
    config = asdict(SyntheticConfig())
    if factor == "n_discrete_parents_R" and level == 0:
        # delta = 1 is the base configuration (a binary discrete variable that is not a parent of R)
        return config
    if factor == "n_discrete_parents_R":
        # delta = 4 ** m configurations of the discrete parents of R
        config.update(
            n_categories=4, n_discrete=max(1, level), n_discrete_parents_R=level
        )
    else:
        config[factor] = level
    return config


def build_units(experiment: str, args: argparse.Namespace) -> list[UnitSpec]:
    """Units of each experiment, with the sizes of ``args`` (see ``parse_args`` for the defaults and ``--quick``)"""
    lw_sizes, gibbs_sizes = args.sample_sizes, args.gibbs_sample_sizes
    sizes = {"lw": lw_sizes, "gibbs": gibbs_sizes}
    common = dict(
        n_reps=args.n_reps,
        n_reference=args.n_reference,
        n_reference_max=max(args.n_reference_max, args.n_reference),
        n_reference_gibbs=args.n_reference_gibbs,
        reference_burn_in=args.reference_burn_in,
        reference_max_rhat=args.reference_max_rhat,
        n_prior=args.n_prior,
        gibbs_options={"n_chains": args.n_chains, "burn_in": args.burn_in},
    )
    base_evidence = asdict(EvidenceSpec())
    all_methods = ["LW", "RBLW", "GS", "RBGS"]
    units: list[UnitSpec] = []
    seeds = [args.seed + 1000 * i for i in range(args.n_networks)]

    if experiment == "E1":
        for i, seed in enumerate(seeds):
            units.append(
                UnitSpec(
                    f"gaussian_{i}",
                    "gaussian",
                    {"check": "gaussian"},
                    asdict(SyntheticConfig(n_nodes=8, frac_clg=1.0, n_discrete=0)),
                    seed,
                    asdict(EvidenceSpec(n_evidence=3, location="R")),
                    ["clg"],
                    args.n_tasks,
                    all_methods,
                    sizes,
                    reference="exact",
                    **common,
                )
            )
            # 5 variables (1 discrete, 2 CKDE, 2 CLG) with 2 evidence variables: at most 3 unobserved continuous ones
            units.append(
                UnitSpec(
                    f"small_{i}",
                    "synthetic",
                    {"check": "grid"},
                    asdict(
                        SyntheticConfig(
                            n_nodes=5, frac_clg=0.4, n_discrete=1, n_train=500
                        )
                    ),
                    seed,
                    asdict(EvidenceSpec(n_evidence=2, location="both")),
                    QUERY_TYPES,
                    args.n_tasks,
                    all_methods,
                    sizes,
                    reference="grid",
                    **common,
                )
            )
            # Bimodal CKDE x with an observed CLG (in R) or CKDE (in S) child y; the query is always x
            for child_type in ["clg", "ckde"]:
                units.append(
                    UnitSpec(
                        f"two_node_{child_type}_{i}",
                        "two_node",
                        {"check": f"two_node_{child_type}"},
                        {"child_type": child_type},
                        seed,
                        asdict(
                            EvidenceSpec(
                                n_evidence=1,
                                location="R" if child_type == "clg" else "S",
                            )
                        ),
                        ["s_continuous"],
                        args.n_tasks,
                        all_methods,
                        sizes,
                        reference="grid",
                        extra={"query_candidates": ["x"]},
                        **common,
                    )
                )
        return units

    if experiment == "E2":
        for i, seed in enumerate(seeds):
            units.append(
                UnitSpec(
                    f"network_{i}",
                    "synthetic",
                    {"factor": "base", "level": "base"},
                    asdict(SyntheticConfig()),
                    seed,
                    base_evidence,
                    QUERY_TYPES,
                    args.n_tasks,
                    all_methods,
                    sizes,
                    reference="lw+gibbs",
                    **common,
                )
            )
        return units

    if experiment == "E3":
        # The base level of every factor is the same base network and tasks, so they are run once ("base" units) and
        # analysis.py uses them as the base level of each factor (REGIME_BASE_LEVELS)
        for i, seed in enumerate(seeds):
            units.append(
                UnitSpec(
                    f"base_{i}",
                    "synthetic",
                    {"factor": "base", "level": "base"},
                    asdict(SyntheticConfig()),
                    seed,
                    base_evidence,
                    ["clg", "s_continuous"],
                    args.n_tasks,
                    all_methods,
                    sizes,
                    reference="lw+gibbs",
                    **common,
                )
            )
        for factor in REGIME_NETWORK_FACTORS:
            for level in NETWORK_FACTORS[factor]:
                if REGIME_BASE_LEVELS.get(factor) == level:
                    continue
                for i, seed in enumerate(seeds):
                    units.append(
                        UnitSpec(
                            f"{factor}={level}_{i}",
                            "synthetic",
                            {"factor": factor, "level": level},
                            _network_factor_config(factor, level),
                            seed,
                            base_evidence,
                            ["clg", "s_continuous"],
                            args.n_tasks,
                            all_methods,
                            sizes,
                            reference="lw+gibbs",
                            **common,
                        )
                    )
        for factor, levels in EVIDENCE_FACTORS.items():
            for level in levels:
                if REGIME_BASE_LEVELS.get(factor) == level:
                    continue
                # Long CLG chains are needed for up to 10 unobserved CLG ancestors
                config = asdict(
                    SyntheticConfig(n_nodes=40 if factor == "n_rb_ancestors" else 20)
                )
                for i, seed in enumerate(seeds):
                    units.append(
                        UnitSpec(
                            f"{factor}={level}_{i}",
                            "synthetic",
                            {"factor": factor, "level": level},
                            config,
                            seed,
                            asdict(replace(EvidenceSpec(), **{factor: level})),
                            ["clg", "s_continuous"],
                            args.n_tasks,
                            all_methods,
                            sizes,
                            reference="lw+gibbs",
                            **common,
                        )
                    )
        return units

    if experiment == "E4":
        for i, seed in enumerate(seeds):
            # (i) Decomposition of the RBLW gain on CLG queries with evidence on R behind unobserved CLG variables
            units.append(
                UnitSpec(
                    f"decomposition_{i}",
                    "synthetic",
                    {"ablation": "decomposition"},
                    asdict(SyntheticConfig()),
                    seed,
                    asdict(EvidenceSpec(n_evidence=4, n_rb_ancestors=2)),
                    ["clg"],
                    args.n_tasks,
                    ["LW", "RBLW-KDE", "RBLW"],
                    sizes,
                    reference="lw+gibbs",
                    **common,
                )
            )
            # (ii) Block moves in a classifier whose classes separate the CKDE features
            units.append(
                UnitSpec(
                    f"block_moves_{i}",
                    "synthetic",
                    {"ablation": "block_moves"},
                    asdict(
                        SyntheticConfig(
                            n_nodes=10,
                            frac_clg=0.3,
                            n_discrete=1,
                            prob_discrete_parent=1.0,
                            class_separation=3.0,
                        )
                    ),
                    seed,
                    asdict(EvidenceSpec(n_evidence=3, location="S")),
                    ["discrete"],
                    args.n_tasks,
                    ["RBGS", "RBGS-noblock"],
                    sizes,
                    reference="lw+gibbs",
                    **common,
                )
            )
            # (iii) Rejection rounds, with evidence on CKDE variables so that the updates of their parents need rejection
            units.append(
                UnitSpec(
                    f"rejection_{i}",
                    "synthetic",
                    {"ablation": "rejection"},
                    asdict(SyntheticConfig()),
                    seed,
                    asdict(EvidenceSpec(n_evidence=4, location="S")),
                    ["s_continuous"],
                    args.n_tasks,
                    ["RBGS-rej0", "RBGS-rej1", "RBGS-rej10", "RBGS-rej100"],
                    sizes,
                    reference="lw+gibbs",
                    **common,
                )
            )
        return units

    if experiment == "E5":
        for factor in SCALING_FACTORS:
            for level in NETWORK_FACTORS[factor]:
                for i, seed in enumerate(seeds):
                    units.append(
                        UnitSpec(
                            f"{factor}={level}_{i}",
                            "synthetic",
                            {"factor": factor, "level": level},
                            _network_factor_config(factor, level),
                            seed,
                            base_evidence,
                            ["clg"],
                            1,
                            all_methods,
                            sizes,
                            reference="none",
                            **common,
                        )
                    )
        return units

    if experiment == "E7":
        for dataset in args.datasets:
            for model_name in CLASSIFIERS:
                units.append(
                    UnitSpec(
                        f"{dataset}_{model_name}",
                        "classifier",
                        {"dataset": dataset, "model": model_name},
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "max_train_size": args.max_train_size,
                        },
                        args.seed,
                        {},
                        ["discrete"],
                        args.n_tasks,
                        all_methods,
                        {
                            "lw": [args.classifier_sample_size],
                            "gibbs": [args.classifier_sample_size],
                        },
                        reference="stratified_lw",
                        extra={"missing_rates": [0.1, 0.2, 0.3, 0.5]},
                        **{**common, "n_reps": args.classifier_reps},
                    )
                )
        return units

    raise ValueError(
        f"Unknown experiment {experiment!r}, expected one of {EXPERIMENT_NAMES}."
    )


# endregion Experiment definitions


# region Execution
STAGE_FUNCTIONS = {
    "references": prepare_unit,
    "runs": run_unit,
    "runs_gpu": run_unit_gpu,
    "runs_cpu": run_unit_cpu,
}


def _run_stage_safely(
    stage: str,
    unit: UnitSpec,
    unit_path: Path,
    shard: tuple[int, int] | None = None,
) -> tuple[str, int, str]:
    """Runs a stage on a unit (or a shard of it), returning the traceback instead of raising."""
    label = (
        unit.unit_id if shard is None else f"{unit.unit_id} [{shard[0] + 1}/{shard[1]}]"
    )
    try:
        arguments = (unit, unit_path) if shard is None else (unit, unit_path, shard)
        return label, STAGE_FUNCTIONS[stage](*arguments), ""
    except Exception:
        return label, 0, traceback.format_exc()


def _safe_name(unit_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_=." else "_" for c in unit_id)


def _execute_stage(
    stage: str,
    units: list[UnitSpec],
    output_path: Path,
    n_workers: int,
    gpus: list[str],
    cores: list[int] | None = None,
    n_shards: int = 1,
) -> list[tuple[str, str]]:
    """Runs a stage on the units, in this process or with ``n_workers`` spawned processes.

    Each worker gets its own GPU (round-robin over ``gpus``) and, with ``cores``, its own CPU core, so that parallel
    timed runs do not compete for the same devices.

    Args:
        stage: A key of ``STAGE_FUNCTIONS``.
        units: The units.
        output_path: The experiment folder.
        n_workers: Parallel processes; with 1 and no ``cores``, the units run in this process.
        gpus: GPUs of the workers.
        cores: CPU cores of the workers (one each), or None to leave them unpinned.
        n_shards: Number of shards of the tasks of each unit (timed stage only), each run as its own job.

    Returns:
        list[tuple[str, str]]: The failed units with their tracebacks.
    """
    label = "references" if stage == "references" else "rows"
    failures = []
    device = stage.removeprefix("runs_")
    jobs = []
    for unit in units:
        unit_path = output_path / "units" / _safe_name(unit.unit_id)
        if n_shards <= 1:
            jobs.append((unit, unit_path, None))
            continue
        for index in range(n_shards):
            if not (unit_path / f"done_{device}_{index}of{n_shards}").exists():
                jobs.append((unit, unit_path, (index, n_shards)))

    def _report(index: int, unit_id: str, count: int, error: str) -> None:
        print(
            f"[{stage} {index + 1}/{len(jobs)}] {unit_id}: {count} {label}"
            + (" FAILED" if error else ""),
            flush=True,
        )
        if error:
            failures.append((unit_id, error))

    if not jobs:
        return failures
    if n_workers <= 1 and not cores:
        for index, (unit, unit_path, shard) in enumerate(jobs):
            _report(index, *_run_stage_safely(stage, unit, unit_path, shard))
        return failures

    n_workers = max(n_workers, 1)
    if stage in ("runs", "runs_gpu") and gpus and n_workers > len(gpus):
        warnings.warn(
            f"{n_workers} workers share {len(gpus)} GPUs: the runtimes of the workers on the same GPU are not independent.",
            RuntimeWarning,
        )
    # spawn: pybnesian/OpenCL contexts must not be inherited through fork
    context = multiprocessing.get_context("spawn")
    slots = context.Queue()
    for slot in worker_slots(n_workers, gpus, cores):
        slots.put(slot)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=context,
        initializer=assign_worker_device,
        initargs=(slots,),
    ) as executor:
        futures = [
            executor.submit(_run_stage_safely, stage, unit, unit_path, shard)
            for unit, unit_path, shard in jobs
        ]
        for index, future in enumerate(as_completed(futures)):
            _report(index, *future.result())
    return failures


def run_experiment(experiment: str, args: argparse.Namespace) -> Path:
    """Runs (or resumes) the units of an experiment in two stages and concatenates their files in the experiment folder.

    - "references": builds the networks, tasks and references. It is not timed, so it can use many workers
      (``--reference_workers``), and it is the most expensive stage.
    - "runs": runs and times the inference methods. With several workers (``--max_workers``), each worker has its own GPU
      and CPU core, so keep at most one worker per GPU.
    ``--stage all`` runs both stages in order.
    """
    output_path = Path(args.output_dir) / experiment
    units = build_units(experiment, args)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "config.json", "w") as f:
        json.dump(
            {"args": vars(args), "units": [asdict(u) for u in units]},
            f,
            default=_json_default,
            indent=2,
        )

    def _pending(*markers: str) -> list[UnitSpec]:
        """Units without any of ``markers``"""
        return [
            u
            for u in units
            if not any(
                (output_path / "units" / _safe_name(u.unit_id) / m).exists()
                for m in markers
            )
        ]

    gpus = args.gpus if args.gpus is not None else available_gpus()
    failures = []
    if args.stage in ("all", "references"):
        pending = _pending("references_done", "done")
        print(
            f"{experiment} references: {len(units) - len(pending)}/{len(units)} units already done, running {len(pending)}."
        )
        failures += _execute_stage(
            "references",
            pending,
            output_path,
            args.reference_workers,
            gpus,
        )
    if args.stage in ("all", "runs"):
        pending = _pending("done")
        repaired = repair_reference_checks(
            [output_path / "units" / _safe_name(u.unit_id) for u in pending]
        )
        if repaired:
            print(f"{experiment}: {repaired} wrongly failed reference checks repaired.")
        print(
            f"{experiment} runs: {len(units) - len(pending)}/{len(units)} units already done, running {len(pending)}."
        )
        # The LW family (GPU workers, one per GPU) and the Gibbs family (CPU workers) run at the same time on disjoint
        # cores. Each pair of methods is timed in the same worker, so the load of the other workers affects both alike,
        all_cores = (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        )
        parts = {
            "gpu": (args.max_workers, all_cores[: args.max_workers]),
            "cpu": (
                args.cpu_workers,
                all_cores[args.max_workers : args.max_workers + args.cpu_workers],
            ),
        }
        with ThreadPoolExecutor(max_workers=len(parts)) as threads:
            jobs = []
            for device, (n_workers, cores) in parts.items():
                part_units = [
                    u
                    for u in pending
                    if device in _unit_devices(u)
                    and not (
                        output_path / "units" / _safe_name(u.unit_id) / f"done_{device}"
                    ).exists()
                ]
                # The shards depend on all the units of the part, not on the pending ones, so a resumed run keeps the
                # shards that are done,
                n_part_units = sum(device in _unit_devices(u) for u in units)
                n_shards = (
                    max(
                        1,
                        -(-SHARDS_PER_CPU_WORKER * n_workers // max(n_part_units, 1)),
                    )
                    if device == "cpu"
                    else 1
                )
                print(
                    f"{experiment} runs on {device}: {len(part_units)} units"
                    + (f", {n_shards} shards each." if n_shards > 1 else ".")
                )
                jobs.append(
                    threads.submit(
                        _execute_stage,
                        f"runs_{device}",
                        part_units,
                        output_path,
                        n_workers,
                        gpus,
                        None if args.no_pin or not cores else cores,
                        n_shards,
                    )
                )
            for job in jobs:
                failures += job.result()
    for unit_id, error in failures:
        print(f"Unit {unit_id} failed:\n{error}")
    combine_units(output_path, [_safe_name(u.unit_id) for u in units])
    return output_path


def combine_units(output_path: Path, unit_names: list[str] | None = None) -> None:
    """Concatenates the files of the finished units into the experiment folder. With ``unit_names``, only those units
    are combined, so the folders of units of an older definition of the experiment are ignored.
    """
    unit_paths = sorted(
        p
        for p in (output_path / "units").glob("*")
        if (p / "done").exists() and (unit_names is None or p.name in unit_names)
    )
    results = [
        pd.read_csv(p / "results.csv")
        for p in unit_paths
        if (p / "results.csv").stat().st_size > 1
    ]
    if results:
        pd.concat(results, ignore_index=True).to_csv(
            output_path / "results.csv", index=False
        )
    networks = []
    for p in unit_paths:
        if (p / "network.json").exists():
            networks.append(json.loads((p / "network.json").read_text()))
    if networks:
        pd.DataFrame(networks).to_csv(output_path / "networks.csv", index=False)
    for name in ["tasks.jsonl", "references.jsonl"]:
        with open(output_path / name, "w") as f:
            for p in unit_paths:
                if (p / name).exists():
                    f.write((p / name).read_text())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", choices=EXPERIMENT_NAMES)
    parser.add_argument(
        "--quick", action="store_true", help="Tiny sizes for a smoke run."
    )
    parser.add_argument(
        "--n_networks", type=int, help="Synthetic networks per configuration."
    )
    parser.add_argument(
        "--n_tasks", type=int, help="Tasks per query type (test instances for E7)."
    )
    parser.add_argument(
        "--n_reps", type=int, help="Repetitions per method and sample size."
    )
    parser.add_argument(
        "--sample_sizes", type=int, nargs="+", help="Sample sizes of LW and RBLW."
    )
    parser.add_argument(
        "--gibbs_sample_sizes", type=int, nargs="+", help="Sample sizes of GS and RBGS."
    )
    parser.add_argument(
        "--n_reference", type=int, help="Samples of the likelihood weighting reference."
    )
    parser.add_argument(
        "--n_reference_max",
        type=int,
        help="Maximum samples of the likelihood weighting reference: it continues past --n_reference while its ESS is "
        f"below {MIN_REFERENCE_ESS}. Defaults to 10 times --n_reference.",
    )
    parser.add_argument(
        "--n_reference_gibbs",
        type=int,
        help="Samples of the check of the reference: Gibbs samples for continuous queries, samples per category of the "
        "stratified likelihood weighting for discrete queries (0 disables the check).",
    )
    parser.add_argument(
        "--reference_burn_in",
        type=int,
        help="Burn-in sweeps of the Gibbs check of the reference.",
    )
    parser.add_argument(
        "--reference_max_rhat",
        type=float,
        help="Maximum R-hat of the Gibbs check of the reference.",
    )
    parser.add_argument(
        "--n_prior",
        type=int,
        help="Prior samples used to draw the evidence and the grid ranges.",
    )
    parser.add_argument("--n_chains", type=int, help="Chains of GS and RBGS.")
    parser.add_argument("--burn_in", type=int, help="Burn-in sweeps of GS and RBGS.")
    parser.add_argument("--datasets", nargs="+", help="Public datasets of E7.")
    parser.add_argument(
        "--max_train_size",
        type=int,
        help="Maximum training instances of the real datasets.",
    )
    parser.add_argument(
        "--classifier_sample_size", type=int, help="Samples per class posterior in E7."
    )
    parser.add_argument(
        "--classifier_reps", type=int, help="Repetitions per test instance in E7."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "references", "runs"],
        default="all",
        help="'references' builds the networks, tasks and references (untimed); 'runs' runs and times the methods "
        "(computing the missing references first); 'all' runs both stages.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        help="Parallel units of the GPU part of the timed stage (LW family); each worker gets its own GPU and CPU core. "
        "Defaults to one worker per GPU (1 without GPUs).",
    )
    parser.add_argument(
        "--cpu_workers",
        type=int,
        help="Parallel units of the CPU part of the timed stage (Gibbs family and imputation), each pinned to its own "
        f"core. Defaults to {TIMED_CORES} minus --max_workers, so the timed stage uses the first {TIMED_CORES} cores.",
    )
    parser.add_argument(
        "--reference_workers",
        type=int,
        help=f"Parallel units of the (untimed) reference stage. Defaults to {DEFAULT_REFERENCE_WORKERS} (at most the "
        "number of CPUs).",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        help="GPUs (indices or UUIDs) for the workers. Defaults to the CUDA_VISIBLE_DEVICES set before the run, or all GPUs.",
    )
    parser.add_argument(
        "--no_pin",
        action="store_true",
        help="Do not pin each worker of the timed stage to its own CPU core.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Defaults to data/results/inference (data/results/inference/quick with --quick, so smoke runs never mark the "
        "units of a full run as done).",
    )
    args = parser.parse_args(argv)
    if args.max_workers is None:
        # One timed worker per GPU, so that no two timed runs share a device
        args.max_workers = max(
            len(args.gpus if args.gpus is not None else available_gpus()), 1
        )
    if args.cpu_workers is None:
        args.cpu_workers = max(
            min(TIMED_CORES, os.cpu_count() or 1) - args.max_workers, 1
        )
    if args.reference_workers is None:
        # The reference stage is not timed, so its workers can share the GPUs
        args.reference_workers = min(DEFAULT_REFERENCE_WORKERS, os.cpu_count() or 1)
    if args.output_dir is None:
        args.output_dir = str(RESULT_PATH / "quick" if args.quick else RESULT_PATH)

    defaults = dict(
        n_networks=10 if args.experiment in ("E1", "E2") else 5,
        n_tasks=5 if args.experiment != "E7" else 50,
        # The statistical unit is the network, so 20 repetitions (RMSE of a task within about 16%) are enough,
        n_reps=20,
        sample_sizes=[100, 1_000, 10_000, 100_000],
        gibbs_sample_sizes=[100, 1_000, 10_000],
        n_reference=10_000_000,
        n_reference_gibbs=200_000,
        reference_burn_in=1000,
        reference_max_rhat=MAX_REFERENCE_RHAT,
        n_prior=20_000,
        n_chains=GIBBS_DEFAULTS["n_chains"],
        burn_in=GIBBS_DEFAULTS["burn_in"],
        datasets=DATASET_NAME_LIST,
        max_train_size=2000,
        classifier_sample_size=1000,
        classifier_reps=5,
    )
    if args.experiment == "E3":
        # Tasks of the same network are correlated, so E3 uses more networks with fewer tasks each (tested per network)
        # 170 units: 10 repetitions and N up to 10**4 (LW family) and 10**3 (Gibbs family) keep it affordable, since
        # the relative RMSE reduction barely depends on N,
        defaults.update(
            n_networks=10,
            n_tasks=3,
            n_reps=10,
            sample_sizes=[100, 1_000, 10_000],
            gibbs_sample_sizes=[100, 1_000],
        )
    if args.experiment == "E7":
        # One stratified reference per test instance, missing rate and classifier: 10**6 samples per class keeps it affordable
        defaults.update(n_reference=1_000_000)
    if args.experiment == "E5":
        defaults.update(
            n_reps=5,
            sample_sizes=[1, 100, 1_000, 10_000, 100_000],
            gibbs_sample_sizes=[100, 1_000, 10_000],
        )
    if args.quick:
        defaults.update(
            n_networks=1,
            n_tasks=1 if args.experiment != "E7" else 2,
            n_reps=2,
            sample_sizes=[100, 1_000],
            gibbs_sample_sizes=[100, 400],
            n_reference=20_000,
            # No check of the reference: the chains of a quick run are too short to converge
            n_reference_gibbs=0,
            reference_burn_in=50,
            reference_max_rhat=1.1,
            n_prior=2_000,
            n_chains=20,
            burn_in=20,
            datasets=["iris"],
            max_train_size=300,
            classifier_sample_size=200,
            classifier_reps=1,
        )
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.n_reference_max is None:
        args.n_reference_max = 10 * args.n_reference
    return args


if __name__ == "__main__":
    arguments = parse_args()
    path = run_experiment(arguments.experiment, arguments)
    print(f"Results in {path}")

# endregion Execution
