"""Experiments of the article on Rao-Blackwellized inference in semiparametric Bayesian networks (sections/04-results.tex).

Each experiment (E1-E7) writes its raw results to ``data/results/inference/<experiment>/``:
    - ``units/<unit_id>/``: the network, tasks, references and results of each independent unit (a network or a dataset),
      written when the unit finishes, so an interrupted run resumes by skipping the finished units.
    - ``results.csv``, ``tasks.jsonl``, ``references.jsonl`` and ``networks.csv``: the concatenation of all the units.
``inference_analysis.py`` builds the figures, tables and statistical tests from these files.

Usage (from the library root):
    python -m src.inference_experiments E2 [--quick] [--max_workers 1] [--n_reps 50] ...
"""

# region Imports
import os

# The OpenCL device must be chosen before pybnesian is imported, and the BLAS threads fixed before numpy is imported
from .opencl_device import opencl_device_name, select_most_available_gpu

select_most_available_gpu()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import json
import multiprocessing
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pybnesian as pbn
from scipy.special import logsumexp
from scipy.stats import norm

from .bn import SemiParametricBayesianNetwork
from .bnc import (
    SemiParametricBayesianNetworkAugmentedNaiveBayes,
    SemiParametricNaiveBayes,
    SemiParametricTreeAugmentedNaiveBayes,
)
from .data_handler import DATASET_NAME_LIST, DataHandler
from .utils.constants import TRUE_CLASS_LABEL

# endregion Imports

# region Constants
LIBRARY_ROOT_PATH = Path(__file__).resolve().parent.parent
RESULT_PATH = LIBRARY_ROOT_PATH / "data" / "results" / "inference"
EXPERIMENT_NAMES = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]

# Inference methods: the arguments of ``SemiParametricBayesianNetwork.infer``, plus the posterior ``estimator``:
# "default" is the estimate of the article (mixture of exact posteriors for the Rao-Blackwellized CLG query, KDE for a
# sampled continuous query, frequencies for a discrete query), and "sampled_kde" draws the Rao-Blackwellized query from its
# exact posterior given each sample and uses a KDE (ablation E4).
METHODS = {
    "LW": {"method": "lw", "rao_blackwellize": False},
    "RBLW": {"method": "lw", "rao_blackwellize": True},
    "GS": {"method": "gibbs", "rao_blackwellize": False},
    "RBGS": {"method": "gibbs", "rao_blackwellize": True},
    # E4 ablations
    "RBLW-KDE": {"method": "lw", "rao_blackwellize": True, "estimator": "sampled_kde"},
    "RBGS-noblock": {"method": "gibbs", "rao_blackwellize": True, "block_moves": False},
    "RBGS-rej0": {"method": "gibbs", "rao_blackwellize": True, "max_rejection_tries": 0},
    "RBGS-rej1": {"method": "gibbs", "rao_blackwellize": True, "max_rejection_tries": 1},
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
GIBBS_DEFAULTS = {"n_chains": 100, "burn_in": 100, "thinning": 1, "max_rejection_tries": 10}
QUERY_TYPES = ["clg", "s_continuous", "discrete"]
CDF_GRID_SIZE = 1024
# Batch size of the reference likelihood weighting
REFERENCE_BATCH_SIZE = 100_000
# Grid points per dimension of the grid reference, by number of unobserved continuous variables
GRID_POINTS = {1: 4001, 2: 401, 3: 101}
MIN_REFERENCE_ESS = 1000
MAX_REFERENCE_RHAT = 1.01
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
# endregion Constants


# region Synthetic semiparametric Bayesian networks
@dataclass(frozen=True)
class SyntheticConfig:
    """Generative configuration of a synthetic SPBN.

    The discrete variables (D0, D1, ...) are roots, followed in topological order by the CKDE variables and by the CLG
    variables. Since the CKDE variables only have discrete and CKDE parents, the sampled set S is exactly the discrete and
    CKDE block, and the Rao-Blackwellized set R the CLG block, so |R| / n = ``frac_clg`` up to rounding.

    Attributes:
        n_nodes: Total number of variables, discrete ones included.
        frac_clg: Fraction of CLG variables |R| / n.
        n_discrete: Number of discrete root variables.
        n_categories: Number of categories of every discrete variable.
        n_discrete_parents_R: Number of discrete variables that are parents of every CLG variable, so the number of
            configurations delta of the discrete parents of R is n_categories ** n_discrete_parents_R.
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
    """Unit-variance noise: Gaussian for the CLG variables, Student-t (4 degrees of freedom) or a two-component mixture for
    the CKDE variables."""
    if gaussian:
        return rng.standard_normal(size)
    if rng.random() < 0.5:
        return rng.standard_t(4, size) / np.sqrt(2.0)
    component = rng.random(size) < 0.5
    return _standardize(np.where(component, -1.0, 1.0) + 0.5 * rng.standard_normal(size))


def generate_synthetic_spbn(
    config: SyntheticConfig, seed: int
) -> tuple[SemiParametricBayesianNetwork, dict]:
    """Generates a synthetic SPBN: a random structure and CPD types (see ``SyntheticConfig``), training data sampled from a
    nonlinear generative process with that structure, and the CPDs fitted to the data with the given types.

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
        candidates = [c for c in ckde_nodes + clg_nodes[:i] if c not in continuous_parents]
        n_extra = rng.integers(
            1 if not continuous_parents and candidates else 0,
            max(config.max_parents - len(continuous_parents), 0) + 1,
        )
        n_extra = min(n_extra, len(candidates))
        continuous_parents += list(rng.choice(candidates, size=n_extra, replace=False))
        parents[node] = discrete_nodes[: config.n_discrete_parents_R] + continuous_parents

    # Data, sampled in topological order
    n = config.n_train
    data: dict[str, np.ndarray] = {}
    # The discrete parents of R follow a balanced design, so every one of their n_categories ** m configurations has about
    # n_train / n_categories ** m training instances (otherwise the CLG-CPDs of empty configurations cannot be fitted)
    m = config.n_discrete_parents_R
    if m > 0:
        configurations = rng.permutation(np.arange(n) % config.n_categories**m)
        for position, node in enumerate(discrete_nodes[:m]):
            data[node] = (configurations // config.n_categories**position) % config.n_categories
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
            coefficients = rng.choice([-1.0, 1.0], len(continuous_parents)) * rng.uniform(
                0.5, 1.5, len(continuous_parents)
            )
            signal = np.column_stack([data[p] for p in continuous_parents]) @ coefficients
            if not is_clg:
                signal = nonlinearities[rng.integers(len(nonlinearities))](signal)
            values = np.sqrt(config.dependence) * _standardize(signal) + np.sqrt(
                1.0 - config.dependence
            ) * noise
        elif is_clg:
            values = noise
        else:
            # Root CKDE variables are bimodal
            values = _standardize(np.where(rng.random(n) < 0.5, -1.5, 1.5) + 0.6 * noise)
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
    """Expected number of kernels of each CKDE-CPD per evaluation: its number of training instances, or, with discrete
    parents, sum_c L_c**2 / sum_c L_c, the size of the configuration c of a sample drawn with the training frequencies."""
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
    rb_discrete_parents = {p for node in rb_nodes for p in bn.parents(node) if p in discrete}
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
        "n_delta": int(np.prod([len(categories[p]) for p in rb_discrete_parents]))
        if rb_discrete_parents
        else 1,
        "mean_ckde_size": float(np.mean(list(sizes.values()))) if sizes else 0.0,
    }


def two_node_spbn(child_type: str, seed: int, n_train: int = 300) -> SemiParametricBayesianNetwork:
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
        n_rb_ancestors: If set, the evidence on R is a single CLG variable with exactly this number of unobserved CLG
            ancestors (its excess ancestors are also observed), and the rest of the evidence is on S.
        rarity: Prior tail probability of the evidence values: 0.5 draws them jointly from the prior, and q < 0.5 sets each
            continuous evidence variable to its prior q or 1 - q quantile (random side).
    """

    n_evidence: int = 4
    location: str = "both"
    n_rb_ancestors: int | None = None
    rarity: float = 0.5


def node_query_type(bn: SemiParametricBayesianNetwork, node: str, sampled: set[str]) -> str:
    if node in bn._discrete_nodes():
        return "discrete"
    return "s_continuous" if node in sampled else "clg"


def unobserved_rb_ancestors(
    bn: SemiParametricBayesianNetwork, evidence: dict, sampled: set[str]
) -> int:
    """Number of unobserved CLG variables of R that are ancestors of the evidence on R, which determines the variance
    reduction of the RBLW weights (with 0, the RBLW and LW weights coincide)"""
    rb_nodes = set(bn._rb_nodes(sampled))
    ancestors = set()
    for node in evidence:
        if node in rb_nodes:
            ancestors |= _ancestors(bn, node) & rb_nodes
    return len(ancestors - set(evidence))


def _ancestors(bn: SemiParametricBayesianNetwork, node: str) -> set[str]:
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
    """Generates up to ``n_tasks`` tasks per query type with the evidence of ``spec``. Infeasible combinations (e.g., no
    unobserved CLG variable left, or no CLG variable with enough ancestors) are skipped.

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
        key: [n for n in topo_order if n in nodes and (query_candidates is None or n in query_candidates)]
        for key, nodes in pools.items()
    }
    tasks = []
    for query_type in query_types:
        for task_index in range(n_tasks):
            task = _draw_task(bn, query_type, spec, rng, prior, pools, sampled, rb_nodes)
            if task is None:
                continue
            task.task_id = f"{id_prefix}{query_type}_{task_index}"
            tasks.append(task)
    return tasks


def _draw_task(bn, query_type, spec, rng, prior, pools, sampled, rb_nodes) -> Task | None:
    if not pools[query_type]:
        return None
    query = str(rng.choice(pools[query_type]))
    s_pool = [str(n) for n in sampled if n != query]
    r_pool = [str(n) for n in rb_nodes if n != query]
    evidence_nodes: list[str] = []
    if spec.n_rb_ancestors is not None:
        rb_set = set(rb_nodes)
        counts = {
            n: len((_ancestors(bn, n) & rb_set) - {query}) for n in r_pool
        }
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
        evidence_nodes += list(rng.choice(s_pool, size=min(n_s, len(s_pool)), replace=False))
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
            "evidence_location": spec.location
            if spec.n_rb_ancestors is None
            else "both",
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


def _evaluation_grid(mean: float, std: float, lo: float, hi: float) -> np.ndarray:
    """Grid of the CDF errors: the reference posterior range widened by 5 posterior standard deviations"""
    return np.linspace(lo - 5.0 * std, hi + 5.0 * std, CDF_GRID_SIZE)


def _reference_from_weighted_samples(
    values: np.ndarray, log_weights: np.ndarray, is_discrete: bool, categories: list[str]
) -> Reference:
    weights = np.exp(log_weights - logsumexp(log_weights))
    ess = float(1.0 / np.sum(weights**2))
    if is_discrete:
        probabilities = {c: float(np.sum(weights[values == c])) for c in categories}
        mcse = float(np.max([np.sqrt(p * (1 - p) / ess) for p in probabilities.values()]))
        return Reference("lw", True, probabilities=probabilities, mcse=mcse, effective_sample_size=ess)
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
    bn: SemiParametricBayesianNetwork, task: Task, n_samples: int, seed: int
) -> Reference:
    """Reference posterior with plain likelihood weighting (every variable sampled), in batches combined through their
    ``log_evidence``, so the unnormalized weights of all the batches are on the same scale"""
    values, log_weights = [], []
    seeds = np.random.SeedSequence(seed).spawn(int(np.ceil(n_samples / REFERENCE_BATCH_SIZE)))
    remaining = n_samples
    for child in seeds:
        batch = min(REFERENCE_BATCH_SIZE, remaining)
        remaining -= batch
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
            log_weights.append(
                np.log(parameters["weights"]) + parameters["log_evidence"] + np.log(batch)
            )
        column = parameters["assignments"][task.query]
        values.append(column.astype(str).to_numpy() if task.query_type == "discrete" else column.to_numpy(dtype=float))
    is_discrete = task.query_type == "discrete"
    categories = bn._categories()[task.query] if is_discrete else []
    return _reference_from_weighted_samples(
        np.concatenate(values), np.concatenate(log_weights), is_discrete, categories
    )


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
            c: float(np.mean(column.astype(str) == c)) for c in bn._categories()[task.query]
        }
        mcse = float(np.max([np.sqrt(p * (1 - p) / ess) for p in probabilities.values()]))
        return Reference("gibbs", converged, probabilities=probabilities, mcse=mcse, effective_sample_size=ess, message=message)
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


def checked_reference(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    n_samples: int,
    n_gibbs_samples: int,
    seed: int,
    gibbs_burn_in: int = 1000,
    max_rhat: float = MAX_REFERENCE_RHAT,
) -> Reference:
    """Likelihood weighting reference checked against an independent reference: the task is kept (``ok``) only if the
    likelihood weighting ESS is at least ``MIN_REFERENCE_ESS``, the check is valid and both agree within 3 combined Monte
    Carlo standard errors. The check uses long plain Gibbs chains (which must converge, R-hat < ``max_rhat``) for continuous
    queries, and stratified likelihood weighting (one run per category, ``n_gibbs_samples`` samples each) for discrete
    queries, since plain Gibbs sampling barely moves between categories that separate their children. With
    ``n_gibbs_samples <= 0`` there is no check (quick runs).
    The Rao-Blackwellized methods are never used, so the reference is independent of the methods that it evaluates."""
    reference = lw_reference(bn, task, n_samples, seed)
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
        check = gibbs_reference(bn, task, n_gibbs_samples, seed + 1, burn_in=gibbs_burn_in, max_rhat=max_rhat)
        reference.method = "lw+gibbs"
    if not check.ok:
        reference.ok = False
        reference.message = check.message
        return reference
    tolerance = 3.0 * np.sqrt(reference.mcse**2 + check.mcse**2)
    if task.query_type == "discrete":
        difference = max(
            abs(reference.probabilities[c] - check.probabilities[c]) for c in reference.probabilities
        )
    else:
        difference = abs(reference.mean - check.mean)
    if not np.isfinite(tolerance) or difference > tolerance:
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
        posterior_mean, posterior_std, posterior_mean - 5 * posterior_std, posterior_mean + 5 * posterior_std
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
    bn: SemiParametricBayesianNetwork, points: pd.DataFrame, discrete_values: dict[str, str]
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
    bn: SemiParametricBayesianNetwork, task: Task, prior: pd.DataFrame, chunk_size: int = 4000
) -> Reference:
    """Reference posterior by numerical integration on a grid of the (at most 3) unobserved continuous variables,
    enumerating the unobserved discrete variables. A first pass on the prior range locates the posterior mass of each
    variable, and a second pass refines the grid on that region."""
    discrete = bn._discrete_nodes()
    categories = bn._categories() if discrete else {}
    unobserved = [n for n in bn.graph().topological_sort() if n not in task.evidence]
    continuous = [n for n in unobserved if n not in discrete]
    hidden_discrete = [n for n in unobserved if n in discrete]
    if len(continuous) > 3:
        raise ValueError("The grid reference supports at most 3 unobserved continuous variables.")
    n_points = GRID_POINTS.get(len(continuous), 1)

    def _evaluate(axes: dict[str, np.ndarray]):
        """Unnormalized posterior on the grid: marginal of the query and of every continuous axis"""
        mesh = np.meshgrid(*(axes[n] for n in continuous), indexing="ij") if continuous else []
        flat = {n: m.ravel() for n, m in zip(continuous, mesh)}
        n_rows = len(next(iter(flat.values()))) if flat else 1
        configurations = list(itertools.product(*(categories[n] for n in hidden_discrete)))
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
                    points[n] = pd.Categorical([v] * (stop - start), categories=categories[n])
                log_density[start:stop] = _joint_log_density(bn, points, fixed)
            log_blocks.append(log_density)
        log_blocks = np.array(log_blocks)  # (configurations, grid points)
        weights = np.exp(log_blocks - np.max(log_blocks))
        return configurations, weights.reshape((len(configurations), *[len(axes[n]) for n in continuous]))

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
            marginal = weights.sum(axis=tuple(i for i in range(weights.ndim) if i != d + 1))
            cumulative = np.cumsum(marginal) / marginal.sum()
            lo_index = max(np.searchsorted(cumulative, 1e-7) - 2, 0)
            hi_index = min(np.searchsorted(cumulative, 1 - 1e-7) + 2, len(axes[node]) - 1)
            refined[node] = np.linspace(axes[node][lo_index], axes[node][hi_index], n_points)
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
    density = density / _trapezoid(density, axis)
    mean = float(_trapezoid(axis * density, axis))
    std = float(np.sqrt(_trapezoid((axis - mean) ** 2 * density, axis)))
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
    """Reference class posterior P(c | e) ∝ f(e, c) of a classifier, with one plain likelihood weighting run per class
    clamped as evidence, whose ``log_evidence`` estimates log f(e, c). Stratifying by class avoids sampling the class."""
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
        (1 - p) ** 2 * p**2 * r[c] + p**2 * sum(probabilities[k] ** 2 * r[k] for k in range(len(r)) if k != c)
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
def _weighted_normal_cdf(
    grid: np.ndarray, weights: np.ndarray, means: np.ndarray, scales: np.ndarray | float, chunk: int = 64
) -> np.ndarray:
    """sum_j w_j Phi((g - m_j) / s_j) at every grid point, in chunks of the grid to bound the memory"""
    cdf = np.empty(len(grid))
    scales = np.broadcast_to(np.asarray(scales, dtype=float), means.shape)
    for start in range(0, len(grid), chunk):
        block = grid[start : start + chunk]
        cdf[start : start + chunk] = norm.cdf(
            (block[:, None] - means[None, :]) / scales[None, :]
        ) @ weights
    return cdf


def mixture_estimate(
    weights: np.ndarray, means: np.ndarray, stds: np.ndarray | float, grid: np.ndarray
) -> dict:
    """Moments and CDF of the Gaussian mixture sum_j w_j N(m_j, s_j**2), the estimate of a Rao-Blackwellized CLG query"""
    stds = np.broadcast_to(np.asarray(stds, dtype=float), means.shape)
    mean = float(np.sum(weights * means))
    std = float(np.sqrt(np.sum(weights * (stds**2 + (means - mean) ** 2))))
    return {"mean": mean, "std": std, "cdf": _weighted_normal_cdf(grid, weights, means, stds)}


def kde_estimate(
    weights: np.ndarray, samples: np.ndarray, grid: np.ndarray, effective_sample_size: float | None = None
) -> dict:
    """Moments and CDF of the weighted Gaussian KDE of the article, with the bandwidth of
    ``SemiParametricBayesianNetwork._kde_bandwidth``; the variance of the KDE is the weighted variance plus h**2"""
    bandwidth = SemiParametricBayesianNetwork._kde_bandwidth(samples, weights, effective_sample_size)
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
    """Posterior estimate of the query (Equation posterior_estimate of the article) and its errors with respect to the
    reference: the mean and standard deviation errors and the Wasserstein-1 distance normalized by the reference standard
    deviation for continuous queries, and the maximum absolute probability error for discrete queries."""
    parameters = infer_dict["parameters"]
    weights = np.asarray(parameters["weights"], dtype=float)
    column = parameters["assignments"][task.query]
    if task.query_type == "discrete":
        values = column.astype(str).to_numpy()
        probabilities = {c: float(np.sum(weights[values == c])) for c in bn._categories()[task.query]}
        row = {"probabilities": json.dumps(probabilities)}
        if reference.ok and reference.probabilities:
            row["prob_error"] = max(
                abs(probabilities[c] - reference.probabilities[c]) for c in probabilities
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
        draws = values + np.asarray(posterior_std[task.query]) * rng.standard_normal(len(values))
        estimate = kde_estimate(weights, draws, grid, ess)
    elif task.query in posterior_std:
        estimate = mixture_estimate(weights, values, posterior_std[task.query], grid)
    else:
        estimate = kde_estimate(weights, values, grid, ess)
    row = {"mean": estimate["mean"], "std": estimate["std"], "bandwidth": estimate.get("bandwidth", np.nan)}
    if reference.ok and len(grid) > 0:
        row["err_mean"] = (estimate["mean"] - reference.mean) / reference.std
        row["err_std"] = (estimate["std"] - reference.std) / reference.std
        row["w1"] = float(_trapezoid(np.abs(estimate["cdf"] - np.asarray(reference.cdf)), grid)) / reference.std
    return row


# endregion Posterior estimates and errors


# region Inference runs
@contextmanager
def count_kernel_evaluations():
    """Counts the kernel evaluations of the CKDE-CPDs computed with the cached mixtures (``_conditional_mixture``, used by
    the Gibbs samplers): the number of rows times the number of kernels of every call. Yields a one-element list."""
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


def run_method(
    bn: SemiParametricBayesianNetwork,
    task: Task,
    method_name: str,
    n_samples: int,
    seed: int,
    reference: Reference,
    sizes: dict[str, float],
    gibbs_options: dict,
    true_value: float | None = None,
) -> dict:
    """Runs one inference method on one task and returns the result row: runtime, kernel evaluations, posterior estimate,
    errors with respect to the reference and diagnostics. With ``true_value`` (E6), the row also contains the log-density
    of the posterior estimate at the true value of the query (``log_score``).

    The repetition seed is shared by all the methods, so likelihood weighting and RBLW draw the same values of the sampled
    variables (common random numbers) and the comparisons are paired.
    """
    spec = dict(METHODS[method_name])
    estimator = spec.pop("estimator", "default")
    kwargs = {"evidence": task.evidence, "n_samples": n_samples, "seed": seed, **spec}
    if spec["method"] == "gibbs":
        kwargs = {**gibbs_options, **kwargs}
    with warnings.catch_warnings(), count_kernel_evaluations() as counter:
        warnings.simplefilter("ignore", RuntimeWarning)
        start = time.perf_counter()
        infer_dict = bn.infer(**kwargs)
        runtime = time.perf_counter() - start

    parameters = infer_dict["parameters"]
    sampled = bn._sampled_nodes(spec["rao_blackwellize"])
    ckde_cost = sum(size for node, size in sizes.items() if node in sampled)
    if spec["method"] == "lw":
        # Every sample evaluates or samples each CKDE-CPD once (all of them are in S)
        kernel_evaluations = n_samples * ckde_cost
    else:
        # The chains are initialized by forward sampling with the CKDE-CPDs of pybnesian
        kernel_evaluations = counter[0] + kwargs["n_chains"] * sum(
            size for node, size in sizes.items() if node in sampled and node not in task.evidence
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
        row["tail_ess"] = parameters["tail_effective_sample_size"].get(task.query, np.nan)
        finite_rhat = [r for r in parameters["rhat"].values() if np.isfinite(r)]
        row["max_rhat"] = max(finite_rhat) if finite_rhat else np.nan
        for rate in ["exact_rate", "acceptance_rate", "block_acceptance_rate"]:
            values = list(parameters[rate].values())
            row[f"mean_{rate}"] = float(np.mean(values)) if values else np.nan
    if true_value is not None:
        ess = parameters["effective_sample_size"].get(task.query) if spec["method"] == "gibbs" else None
        density = bn._posterior_density(
            task.query,
            pd.Series({task.query: true_value}),
            parameters["assignments"],
            parameters["weights"],
            parameters["posterior_std"],
            ess,
        )
        row["log_score"] = float(np.log(max(density, 1e-300)))
    return row


# endregion Inference runs


# region Experiment units
@dataclass
class UnitSpec:
    """An independent unit of an experiment: a network (synthetic or learned from a dataset) and its tasks and runs.

    Attributes:
        unit_id: Unique identifier within the experiment (also the folder name).
        kind: "synthetic", "gaussian", "two_node", "dataset" or "classifier".
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
                    bn, task, unit.n_reference, unit.n_reference_gibbs, unit.seed + 7919,
                    unit.reference_burn_in, unit.reference_max_rhat,
                )
            case "lw":
                return checked_reference(bn, task, unit.n_reference, 0, unit.seed + 7919)
            case "none":
                return Reference("none", False)
    except Exception as error:  # e.g., singular covariance or numerical failure: the task is excluded
        return Reference(unit.reference, False, message=f"{type(error).__name__}: {error}")
    raise ValueError(f"Unknown reference {unit.reference!r}.")


def run_unit(unit: UnitSpec, unit_path: Path) -> int:
    """Runs a unit and writes its files to ``unit_path``; the ``done`` marker is written last, so an interrupted unit is
    run again from scratch. Returns the number of result rows."""
    if (unit_path / "done").exists():
        return -1
    unit_path.mkdir(parents=True, exist_ok=True)
    if unit.kind == "classifier":
        rows = run_classifier_unit(unit, unit_path)
        (unit_path / "done").touch()
        return rows

    test = None
    if unit.kind == "dataset":
        bn, info, test = learn_dataset_spbn(
            unit.config["dataset"], unit.seed, unit.config.get("max_train_size", 2000)
        )
    else:
        bn, info = build_network(unit)
    info.update({"unit_id": unit.unit_id, "kind": unit.kind, **unit.factors})
    info["device"] = opencl_device_name()
    sizes = ckde_sizes(bn)
    prior = bn.sample(unit.n_prior, seed=unit.seed)
    tasks = []
    if unit.kind == "dataset":
        tasks = dataset_tasks(bn, unit, test)
    else:
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
    gibbs_options = {**GIBBS_DEFAULTS, **unit.gibbs_options}

    result_rows, task_rows, reference_rows = [], [], []
    for task in tasks:
        reference = compute_reference(bn, task, unit, prior)
        task_rows.append(asdict(task))
        reference_rows.append({"task_id": task.task_id, **asdict(reference)})
        if unit.reference != "none" and not reference.ok:
            continue
        base = {
            "unit_id": unit.unit_id,
            "task_id": task.task_id,
            "query": task.query,
            "query_type": task.query_type,
            "reference_method": reference.method,
            **unit.factors,
            **task.factors,
            **{key: info[key] for key in ["n_nodes", "n_rb", "frac_rb", "n_delta", "mean_ckde_size"]},
        }
        for method_name in unit.methods:
            family = METHODS[method_name]["method"]
            for n_samples in unit.sample_sizes[family]:
                for rep in range(unit.n_reps):
                    row = run_method(
                        bn, task, method_name, n_samples, rep, reference, sizes, gibbs_options,
                        true_value=task.true_value if unit.kind == "dataset" else None,
                    )
                    result_rows.append({**base, **row})

    pd.DataFrame(result_rows).to_csv(unit_path / "results.csv", index=False)
    with open(unit_path / "tasks.jsonl", "w") as f:
        for row in task_rows:
            f.write(json.dumps(row, default=_json_default) + "\n")
    with open(unit_path / "references.jsonl", "w") as f:
        for row in reference_rows:
            f.write(json.dumps(row, default=_json_default) + "\n")
    with open(unit_path / "network.json", "w") as f:
        json.dump(info, f, default=_json_default, indent=2)
    (unit_path / "done").touch()
    return len(result_rows)


# endregion Experiment units


# region Real datasets (E6, E7)
def load_dataset(dataset_name: str, seed: int, max_train_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test partitions of a public dataset (continuous features and ``TRUE_CLASS_LABEL``)"""
    handler = DataHandler.from_public_data(dataset_name, seed=seed, max_train_data_size=max_train_size)
    train = handler.train_attack_data.reset_index(drop=True)
    test = handler.test_data_dict["test_full"].reset_index(drop=True)
    features = [c for c in train.columns if c != TRUE_CLASS_LABEL]
    train[features] = train[features].astype(float)
    test[features] = test[features].astype(float)
    return train, test


def learn_dataset_spbn(
    dataset_name: str, seed: int, max_train_size: int
) -> tuple[SemiParametricBayesianNetwork, dict, pd.DataFrame]:
    """Learns an SPBN over the continuous features of a dataset with hill climbing and the validated likelihood score.
    Returns the SPBN, its description and the held-out instances of its variables."""
    train, test = load_dataset(dataset_name, seed, max_train_size)
    features = [c for c in train.columns if c != TRUE_CLASS_LABEL]
    bn = SemiParametricBayesianNetwork(seed=seed, max_train_data_size=max_train_size)
    bn.fit(train[features])
    info = describe_network(bn)
    info.update({"dataset": dataset_name, "n_train": len(train), "n_test": len(test)})
    return bn, info, test[list(bn.nodes())]


def dataset_tasks(bn: SemiParametricBayesianNetwork, unit: UnitSpec, test: pd.DataFrame) -> list[Task]:
    """E6 tasks: held-out instances with a random fraction of the variables observed and one of the rest as query"""
    rng = np.random.default_rng(unit.seed + 1)
    sampled = bn._sampled_nodes()
    nodes = list(bn.nodes())
    observed_fraction = unit.extra.get("observed_fraction", 0.5)
    tasks = []
    for task_index, row_index in enumerate(rng.choice(len(test), size=min(unit.n_tasks, len(test)), replace=False)):
        row = test.iloc[int(row_index)]
        n_observed = max(1, min(len(nodes) - 1, int(round(observed_fraction * len(nodes)))))
        permutation = list(rng.permutation(nodes))
        evidence_nodes, query = permutation[:n_observed], permutation[n_observed]
        evidence = {n: float(row[n]) for n in evidence_nodes}
        tasks.append(
            Task(
                task_id=f"{unit.unit_id}/{task_index}",
                query=query,
                query_type=node_query_type(bn, query, sampled),
                evidence=evidence,
                factors={
                    "n_evidence": len(evidence),
                    "n_evidence_S": sum(n in sampled for n in evidence),
                    "n_evidence_R": sum(n not in sampled for n in evidence),
                    "n_unobserved_rb_ancestors": unobserved_rb_ancestors(bn, evidence, sampled),
                },
                true_value=float(row[query]),
            )
        )
    return tasks


CLASSIFIERS = {
    "SP-NB": SemiParametricNaiveBayes,
    "SP-TAN": SemiParametricTreeAugmentedNaiveBayes,
    "SP-BAN": SemiParametricBayesianNetworkAugmentedNaiveBayes,
}


def run_classifier_unit(unit: UnitSpec, unit_path: Path) -> int:
    """E7: class posterior of an SPBN classifier with features missing completely at random, estimated with the inference
    methods, and with mean and k-nearest-neighbors imputation followed by the exact class posterior of the classifier."""
    from sklearn.impute import KNNImputer

    dataset_name, model_name = unit.config["dataset"], unit.config["model"]
    train, test = load_dataset(dataset_name, unit.seed, unit.config.get("max_train_size", 2000))
    features = [c for c in train.columns if c != TRUE_CLASS_LABEL]
    model = CLASSIFIERS[model_name](seed=unit.seed)
    model.fit(train[features], train[TRUE_CLASS_LABEL])
    model_features = list(model.feature_names_in_)
    # Zero-variance features are not variables of the network, so they cannot be evidence
    network_features = [f for f in model_features if f in model.nodes()]
    classes = [str(c) for c in model.classes_]
    info = describe_network(model)
    info.update({"unit_id": unit.unit_id, "dataset": dataset_name, "model": model_name, **unit.factors})
    info["device"] = opencl_device_name()
    sizes = ckde_sizes(model)
    gibbs_options = {**GIBBS_DEFAULTS, **unit.gibbs_options}
    imputers = {
        "MeanImpute": train[model_features].mean(),
        "kNNImpute": KNNImputer(n_neighbors=5).fit(train[model_features]),
    }

    rng = np.random.default_rng(unit.seed + 1)
    rows, task_rows, reference_rows = [], [], []
    test_indices = rng.choice(len(test), size=min(unit.n_tasks, len(test)), replace=False)
    for missing_rate in unit.extra["missing_rates"]:
        for task_index, row_index in enumerate(test_indices):
            row = test.iloc[int(row_index)]
            missing = rng.random(len(network_features)) < missing_rate
            if missing.all():
                missing[rng.integers(len(missing))] = False
            if not missing.any():
                missing[rng.integers(len(missing))] = True
            evidence = {f: float(row[f]) for f, m in zip(network_features, missing) if not m}
            task = Task(
                task_id=f"{unit.unit_id}/{missing_rate}/{task_index}",
                query=model.true_label,
                query_type="discrete",
                evidence=evidence,
                factors={"missing_rate": missing_rate, "n_missing": int(missing.sum())},
                true_value=str(row[TRUE_CLASS_LABEL]),
            )
            reference = stratified_class_reference(model, task, unit.n_reference, unit.seed + 7919)
            task_rows.append(asdict(task))
            reference_rows.append({"task_id": task.task_id, **asdict(reference)})
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
                        max(abs(probabilities.get(c, 0.0) - reference.probabilities[c]) for c in classes)
                    )
                return scores

            for method_name in unit.methods:
                family = METHODS[method_name]["method"]
                for n_samples in unit.sample_sizes[family]:
                    for rep in range(unit.n_reps):
                        result = run_method(model, task, method_name, n_samples, rep, reference, sizes, gibbs_options)
                        probabilities = json.loads(result.pop("probabilities"))
                        result.pop("prob_error", None)
                        rows.append({**base, **result, **_scores(probabilities)})
            for imputer_name, imputer in imputers.items():
                x = pd.DataFrame([row[model_features].to_numpy(dtype=float)], columns=model_features)
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

    pd.DataFrame(rows).to_csv(unit_path / "results.csv", index=False)
    with open(unit_path / "tasks.jsonl", "w") as f:
        for row in task_rows:
            f.write(json.dumps(row, default=_json_default) + "\n")
    with open(unit_path / "references.jsonl", "w") as f:
        for row in reference_rows:
            f.write(json.dumps(row, default=_json_default) + "\n")
    with open(unit_path / "network.json", "w") as f:
        json.dump(info, f, default=_json_default, indent=2)
    return len(rows)


# endregion Real datasets (E6, E7)


# region Experiment definitions
# Levels of the regime factors of Table tab:regime_factors, varied one at a time from the base configuration
NETWORK_FACTORS = {
    "frac_clg": [0.1, 0.3, 0.5, 0.7, 0.9],
    "dependence": [0.3, 0.6, 0.9],
    "n_train": [200, 1000, 5000],
    "n_discrete_parents_R": [0, 1, 2, 3],
    "n_nodes": [10, 20, 50, 100],
}
EVIDENCE_FACTORS = {
    "location": ["S", "both", "R"],
    "n_rb_ancestors": [0, 2, 5, 10],
    "rarity": [0.5, 0.1, 0.01, 0.001],
}


def _network_factor_config(factor: str, level) -> dict:
    config = asdict(SyntheticConfig())
    if factor == "n_discrete_parents_R":
        # delta = 4 ** m configurations of the discrete parents of R
        config.update(n_categories=4, n_discrete=max(1, level), n_discrete_parents_R=level)
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
                    f"gaussian_{i}", "gaussian", {"check": "gaussian"},
                    asdict(SyntheticConfig(n_nodes=8, frac_clg=1.0, n_discrete=0)),
                    seed, asdict(EvidenceSpec(n_evidence=3, location="R")), ["clg"],
                    args.n_tasks, all_methods, sizes, reference="exact", **common,
                )
            )
            # 5 variables (1 discrete, 2 CKDE, 2 CLG) with 2 evidence variables: at most 3 unobserved continuous ones
            units.append(
                UnitSpec(
                    f"small_{i}", "synthetic", {"check": "grid"},
                    asdict(SyntheticConfig(n_nodes=5, frac_clg=0.4, n_discrete=1, n_train=500)),
                    seed, asdict(EvidenceSpec(n_evidence=2, location="both")), QUERY_TYPES,
                    args.n_tasks, all_methods, sizes, reference="grid", **common,
                )
            )
            # Bimodal CKDE x with an observed CLG (in R) or CKDE (in S) child y; the query is always x
            for child_type in ["clg", "ckde"]:
                units.append(
                    UnitSpec(
                        f"two_node_{child_type}_{i}", "two_node", {"check": f"two_node_{child_type}"},
                        {"child_type": child_type}, seed,
                        asdict(EvidenceSpec(n_evidence=1, location="R" if child_type == "clg" else "S")),
                        ["s_continuous"], args.n_tasks, all_methods, sizes, reference="grid",
                        extra={"query_candidates": ["x"]}, **common,
                    )
                )
        return units

    if experiment == "E2":
        for i, seed in enumerate(seeds):
            units.append(
                UnitSpec(
                    f"network_{i}", "synthetic", {"factor": "base", "level": "base"},
                    asdict(SyntheticConfig()), seed, base_evidence, QUERY_TYPES, args.n_tasks,
                    all_methods, sizes, reference="lw+gibbs", **common,
                )
            )
        return units

    if experiment == "E3":
        for factor, levels in NETWORK_FACTORS.items():
            for level in levels:
                for i, seed in enumerate(seeds):
                    units.append(
                        UnitSpec(
                            f"{factor}={level}_{i}", "synthetic", {"factor": factor, "level": level},
                            _network_factor_config(factor, level), seed, base_evidence,
                            ["clg", "s_continuous"], args.n_tasks, all_methods, sizes,
                            reference="lw+gibbs", **common,
                        )
                    )
        for factor, levels in EVIDENCE_FACTORS.items():
            for level in levels:
                # Long CLG chains are needed for up to 10 unobserved CLG ancestors
                config = asdict(SyntheticConfig(n_nodes=40 if factor == "n_rb_ancestors" else 20))
                for i, seed in enumerate(seeds):
                    units.append(
                        UnitSpec(
                            f"{factor}={level}_{i}", "synthetic", {"factor": factor, "level": level},
                            config, seed, asdict(replace(EvidenceSpec(), **{factor: level})),
                            ["clg", "s_continuous"], args.n_tasks, all_methods, sizes,
                            reference="lw+gibbs", **common,
                        )
                    )
        return units

    if experiment == "E4":
        for i, seed in enumerate(seeds):
            # (i) Decomposition of the RBLW gain on CLG queries with evidence on R behind unobserved CLG variables
            units.append(
                UnitSpec(
                    f"decomposition_{i}", "synthetic", {"ablation": "decomposition"},
                    asdict(SyntheticConfig()), seed, asdict(EvidenceSpec(n_evidence=4, n_rb_ancestors=2)),
                    ["clg"], args.n_tasks, ["LW", "RBLW-KDE", "RBLW"], sizes, reference="lw+gibbs", **common,
                )
            )
            # (ii) Block moves in a classifier whose classes separate the CKDE features
            units.append(
                UnitSpec(
                    f"block_moves_{i}", "synthetic", {"ablation": "block_moves"},
                    asdict(SyntheticConfig(n_nodes=10, frac_clg=0.3, n_discrete=1, prob_discrete_parent=1.0, class_separation=3.0)),
                    seed, asdict(EvidenceSpec(n_evidence=3, location="S")), ["discrete"], args.n_tasks,
                    ["RBGS", "RBGS-noblock"], sizes, reference="lw+gibbs", **common,
                )
            )
            # (iii) Rejection rounds, with evidence on CKDE variables so that the updates of their parents need rejection
            units.append(
                UnitSpec(
                    f"rejection_{i}", "synthetic", {"ablation": "rejection"},
                    asdict(SyntheticConfig()), seed, asdict(EvidenceSpec(n_evidence=4, location="S")),
                    ["s_continuous"], args.n_tasks,
                    ["RBGS-rej0", "RBGS-rej1", "RBGS-rej10", "RBGS-rej100"], sizes, reference="lw+gibbs", **common,
                )
            )
        return units

    if experiment == "E5":
        for factor, levels in NETWORK_FACTORS.items():
            if factor == "dependence":
                continue
            for level in levels:
                for i, seed in enumerate(seeds):
                    units.append(
                        UnitSpec(
                            f"{factor}={level}_{i}", "synthetic", {"factor": factor, "level": level},
                            _network_factor_config(factor, level), seed, base_evidence, ["clg"], 1,
                            all_methods, sizes, reference="none", **common,
                        )
                    )
        return units

    if experiment == "E6":
        for dataset in args.datasets:
            units.append(
                UnitSpec(
                    dataset, "dataset", {"dataset": dataset},
                    {"dataset": dataset, "max_train_size": args.max_train_size}, args.seed, {}, [],
                    args.n_tasks, all_methods, sizes, reference="lw+gibbs",
                    extra={"observed_fraction": 0.5}, **common,
                )
            )
        return units

    if experiment == "E7":
        for dataset in args.datasets:
            for model_name in CLASSIFIERS:
                units.append(
                    UnitSpec(
                        f"{dataset}_{model_name}", "classifier", {"dataset": dataset, "model": model_name},
                        {"dataset": dataset, "model": model_name, "max_train_size": args.max_train_size},
                        args.seed, {}, ["discrete"], args.n_tasks, all_methods,
                        {"lw": [args.classifier_sample_size], "gibbs": [args.classifier_sample_size]},
                        reference="stratified_lw", extra={"missing_rates": [0.1, 0.2, 0.3, 0.5]},
                        **{**common, "n_reps": args.classifier_reps},
                    )
                )
        return units

    raise ValueError(f"Unknown experiment {experiment!r}, expected one of {EXPERIMENT_NAMES}.")


# endregion Experiment definitions


# region Execution
def _run_unit_safely(unit: UnitSpec, unit_path: Path) -> tuple[str, int, str]:
    try:
        return unit.unit_id, run_unit(unit, unit_path), ""
    except Exception:
        return unit.unit_id, 0, traceback.format_exc()


def _safe_name(unit_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_=." else "_" for c in unit_id)


def run_experiment(experiment: str, args: argparse.Namespace) -> Path:
    """Runs (or resumes) all the units of an experiment and concatenates their files in the experiment folder"""
    output_path = Path(args.output_dir) / experiment
    units = build_units(experiment, args)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "config.json", "w") as f:
        json.dump({"args": vars(args), "units": [asdict(u) for u in units]}, f, default=_json_default, indent=2)

    pending = [u for u in units if not (output_path / "units" / _safe_name(u.unit_id) / "done").exists()]
    print(f"{experiment}: {len(units) - len(pending)}/{len(units)} units already done, running {len(pending)}.")
    failures = []
    if args.max_workers <= 1:
        for index, unit in enumerate(pending):
            unit_id, n_rows, error = _run_unit_safely(unit, output_path / "units" / _safe_name(unit.unit_id))
            print(f"[{index + 1}/{len(pending)}] {unit_id}: {n_rows} rows" + (" FAILED" if error else ""))
            if error:
                failures.append((unit_id, error))
    else:
        # spawn: pybnesian/OpenCL contexts must not be inherited through fork
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=context) as executor:
            futures = [
                executor.submit(_run_unit_safely, unit, output_path / "units" / _safe_name(unit.unit_id))
                for unit in pending
            ]
            for index, future in enumerate(as_completed(futures)):
                unit_id, n_rows, error = future.result()
                print(f"[{index + 1}/{len(pending)}] {unit_id}: {n_rows} rows" + (" FAILED" if error else ""))
                if error:
                    failures.append((unit_id, error))
    for unit_id, error in failures:
        print(f"Unit {unit_id} failed:\n{error}")
    combine_units(output_path)
    return output_path


def combine_units(output_path: Path) -> None:
    """Concatenates the files of the finished units into the experiment folder"""
    unit_paths = sorted(p for p in (output_path / "units").glob("*") if (p / "done").exists())
    results = [pd.read_csv(p / "results.csv") for p in unit_paths if (p / "results.csv").stat().st_size > 1]
    if results:
        pd.concat(results, ignore_index=True).to_csv(output_path / "results.csv", index=False)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=EXPERIMENT_NAMES)
    parser.add_argument("--quick", action="store_true", help="Tiny sizes for a smoke run.")
    parser.add_argument("--n_networks", type=int, help="Synthetic networks per configuration.")
    parser.add_argument("--n_tasks", type=int, help="Tasks per query type (test instances for E6/E7).")
    parser.add_argument("--n_reps", type=int, help="Repetitions per method and sample size.")
    parser.add_argument("--sample_sizes", type=int, nargs="+", help="Sample sizes of LW and RBLW.")
    parser.add_argument("--gibbs_sample_sizes", type=int, nargs="+", help="Sample sizes of GS and RBGS.")
    parser.add_argument("--n_reference", type=int, help="Samples of the likelihood weighting reference.")
    parser.add_argument(
        "--n_reference_gibbs",
        type=int,
        help="Samples of the check of the reference: Gibbs samples for continuous queries, samples per category of the "
        "stratified likelihood weighting for discrete queries (0 disables the check).",
    )
    parser.add_argument("--reference_burn_in", type=int, help="Burn-in sweeps of the Gibbs check of the reference.")
    parser.add_argument("--reference_max_rhat", type=float, help="Maximum R-hat of the Gibbs check of the reference.")
    parser.add_argument("--n_prior", type=int, help="Prior samples used to draw the evidence and the grid ranges.")
    parser.add_argument("--n_chains", type=int, help="Chains of GS and RBGS.")
    parser.add_argument("--burn_in", type=int, help="Burn-in sweeps of GS and RBGS.")
    parser.add_argument("--datasets", nargs="+", help="Public datasets of E6 and E7.")
    parser.add_argument("--max_train_size", type=int, help="Maximum training instances of the real datasets.")
    parser.add_argument("--classifier_sample_size", type=int, help="Samples per class posterior in E7.")
    parser.add_argument("--classifier_reps", type=int, help="Repetitions per test instance in E7.")
    parser.add_argument("--max_workers", type=int, default=1, help="Parallel units (keep 1 when timing).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Defaults to data/results/inference (data/results/inference/quick with --quick, so smoke runs never mark the "
        "units of a full run as done).",
    )
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = str(RESULT_PATH / "quick" if args.quick else RESULT_PATH)

    defaults = dict(
        n_networks=10 if args.experiment in ("E1", "E2") else 5,
        n_tasks=5 if args.experiment not in ("E6", "E7") else 50,
        n_reps=50 if args.experiment in ("E1", "E2", "E4") else 20,
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
    if args.experiment == "E7":
        # One stratified reference per test instance, missing rate and classifier: 10**6 samples per class keeps it affordable
        defaults.update(n_reference=1_000_000)
    if args.experiment == "E5":
        defaults.update(n_reps=5, sample_sizes=[1, 100, 1_000, 10_000, 100_000], gibbs_sample_sizes=[100, 1_000, 10_000])
    if args.quick:
        defaults.update(
            n_networks=1,
            n_tasks=1 if args.experiment not in ("E6", "E7") else 2,
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
    return args


if __name__ == "__main__":
    arguments = parse_args()
    path = run_experiment(arguments.experiment, arguments)
    print(f"Results in {path}")

# endregion Execution
