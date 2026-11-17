"""Tests of the inference experiments (src/inference_experiments.py) and of their analysis (src/inference_analysis.py)."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pybnesian as pbn
import pytest
from scipy.stats import norm
from src.bn import SemiParametricBayesianNetwork
from src.inference_analysis import (
    analyze_comparison,
    holm_correction,
    task_summary,
    time_to_epsilon,
)
from src.inference_experiments import (
    EvidenceSpec,
    SyntheticConfig,
    Task,
    UnitSpec,
    checked_reference,
    gaussian_reference,
    generate_synthetic_spbn,
    generate_tasks,
    grid_reference,
    kde_estimate,
    lw_reference,
    mixture_estimate,
    run_unit,
    stratified_class_reference,
)

SEED = 0


@pytest.mark.parametrize("frac_clg", [0.3, 0.5, 0.9])
def test_generator_partition(frac_clg: float) -> None:
    """Test that the CKDE and discrete variables form exactly the sampled set, so |R| / n is frac_clg."""
    config = SyntheticConfig(n_nodes=10, frac_clg=frac_clg, n_train=300)
    bn, info = generate_synthetic_spbn(config, SEED)
    assert info["n_rb"] == round(frac_clg * 10)
    assert info["n_sampled"] == 10 - info["n_rb"]
    for node in bn._rb_nodes(bn._sampled_nodes()):
        assert bn.cpd(node).type() == pbn.LinearGaussianCPDType()
    for node in bn._sampled_nodes():
        assert bn.cpd(node).type() in (pbn.CKDEType(), pbn.DiscreteFactorType())


def test_generator_discrete_parents_of_R() -> None:
    """Test the number of configurations delta of the discrete parents of R."""
    config = SyntheticConfig(n_nodes=10, n_discrete=2, n_categories=4, n_discrete_parents_R=2, n_train=500)
    _, info = generate_synthetic_spbn(config, SEED)
    assert info["n_delta"] == 16
    # 64 configurations: the balanced design gives every configuration training data, so plain LW samples finite values
    config = SyntheticConfig(n_nodes=10, n_discrete=3, n_categories=4, n_discrete_parents_R=3, n_train=1000)
    bn, info = generate_synthetic_spbn(config, SEED)
    assert info["n_delta"] == 64
    samples = bn.sample(2000, seed=SEED)
    assert np.all(np.isfinite(samples[bn._rb_nodes(bn._sampled_nodes())].to_numpy(dtype=float)))


def test_task_evidence_spec() -> None:
    """Test the evidence location and the number of unobserved CLG ancestors of the evidence on R."""
    bn, _ = generate_synthetic_spbn(SyntheticConfig(n_nodes=20, chain_prob=1.0, n_train=300), SEED)
    prior = bn.sample(500, seed=SEED)
    sampled = bn._sampled_nodes()
    for location in ["S", "R"]:
        tasks = generate_tasks(bn, 3, ["clg"], EvidenceSpec(n_evidence=2, location=location), SEED, prior)
        for task in tasks:
            assert all((node in sampled) == (location == "S") for node in task.evidence)
            assert task.query not in task.evidence
    tasks = generate_tasks(bn, 3, ["s_continuous"], EvidenceSpec(n_evidence=3, n_rb_ancestors=2), SEED, prior)
    assert tasks and all(task.factors["n_unobserved_rb_ancestors"] == 2 for task in tasks)


def test_grid_reference_matches_gaussian() -> None:
    """Test that the grid reference of a Gaussian network matches its exact posterior."""
    bn, _ = generate_synthetic_spbn(SyntheticConfig(n_nodes=4, frac_clg=1.0, n_discrete=0, n_train=500), SEED)
    prior = bn.sample(2000, seed=SEED)
    tasks = generate_tasks(bn, 1, ["clg"], EvidenceSpec(n_evidence=1, location="R"), SEED, prior)
    exact = gaussian_reference(bn, tasks[0])
    grid = grid_reference(bn, tasks[0], prior)
    np.testing.assert_allclose(grid.mean, exact.mean, atol=1e-3 * exact.std)
    np.testing.assert_allclose(grid.std, exact.std, rtol=1e-3)
    cdf = np.interp(exact.grid, grid.grid, grid.cdf)
    np.testing.assert_allclose(cdf, exact.cdf, atol=2e-3)


def test_grid_reference_matches_lw() -> None:
    """Test that the grid reference of a small SPBN agrees with a large plain likelihood weighting reference."""
    config = SyntheticConfig(n_nodes=5, frac_clg=0.4, n_discrete=1, n_train=300)
    bn, _ = generate_synthetic_spbn(config, SEED)
    prior = bn.sample(2000, seed=SEED)
    tasks = generate_tasks(bn, 1, ["clg", "s_continuous", "discrete"], EvidenceSpec(n_evidence=2), SEED, prior)
    assert tasks
    for task in tasks:
        grid = grid_reference(bn, task, prior)
        lw = lw_reference(bn, task, 200_000, SEED)
        if task.query_type == "discrete":
            for category, probability in grid.probabilities.items():
                assert abs(probability - lw.probabilities[category]) < 5 * lw.mcse + 1e-3
        else:
            assert abs(grid.mean - lw.mean) < 5 * lw.mcse + 1e-3


def test_discrete_reference_check() -> None:
    """Test that a discrete query is checked with stratified likelihood weighting, which agrees with the grid reference and
    has a finite Monte Carlo standard error."""
    config = SyntheticConfig(n_nodes=5, frac_clg=0.4, n_discrete=1, n_train=300)
    bn, _ = generate_synthetic_spbn(config, SEED)
    prior = bn.sample(2000, seed=SEED)
    task = generate_tasks(bn, 1, ["discrete"], EvidenceSpec(n_evidence=2), SEED, prior)[0]
    reference = checked_reference(bn, task, 50_000, 20_000, SEED)
    assert reference.method == "lw+stratified" and reference.ok, reference.message
    stratified = stratified_class_reference(bn, task, 20_000, SEED)
    assert np.isfinite(stratified.mcse) and stratified.mcse > 0
    grid = grid_reference(bn, task, prior)
    for category, probability in grid.probabilities.items():
        assert abs(stratified.probabilities[category] - probability) < 5 * stratified.mcse + 1e-3


def test_posterior_estimates() -> None:
    """Test the moments and CDFs of the mixture and KDE estimates, and that the KDE uses the bandwidth of the network."""
    grid = np.linspace(-6, 6, 2001)
    single = mixture_estimate(np.array([1.0]), np.array([0.5]), 2.0, grid)
    np.testing.assert_allclose(single["cdf"], norm.cdf(grid, 0.5, 2.0), atol=1e-12)
    weights = np.array([0.3, 0.7])
    means = np.array([-1.0, 2.0])
    mixture = mixture_estimate(weights, means, np.array([0.5, 1.0]), grid)
    mean = np.sum(weights * means)
    np.testing.assert_allclose(mixture["mean"], mean)
    np.testing.assert_allclose(mixture["std"] ** 2, np.sum(weights * (np.array([0.25, 1.0]) + (means - mean) ** 2)))

    rng = np.random.default_rng(SEED)
    samples = rng.normal(size=500)
    uniform = np.full(500, 1 / 500)
    kde = kde_estimate(uniform, samples, grid)
    assert kde["bandwidth"] == SemiParametricBayesianNetwork._kde_bandwidth(samples, uniform)
    np.testing.assert_allclose(kde["std"] ** 2, samples.var() + kde["bandwidth"] ** 2)
    assert kde["cdf"][0] < 1e-6 and kde["cdf"][-1] > 1 - 1e-6


def test_holm_and_time_to_epsilon() -> None:
    """Test the Holm correction and the log-log interpolation of the time to reach a target error."""
    np.testing.assert_allclose(holm_correction(np.array([0.01, 0.04, 0.03, np.nan])), [0.03, 0.06, 0.06, np.nan])
    curve = pd.DataFrame({"runtime": [1.0, 10.0, 1000.0], "rmse": [1.0, 1 / np.sqrt(10), 1 / np.sqrt(1000)]})
    np.testing.assert_allclose(time_to_epsilon(curve, 0.1), 100.0, rtol=1e-10)
    assert time_to_epsilon(curve, 1e-3) == np.inf
    assert time_to_epsilon(curve, 2.0) == 0.0


def _small_unit(unit_id: str, reference: str = "lw") -> UnitSpec:
    return UnitSpec(
        unit_id=unit_id,
        kind="synthetic",
        factors={"factor": "base", "level": "base"},
        config=asdict(SyntheticConfig(n_nodes=8, n_train=200)),
        seed=SEED,
        evidence=asdict(EvidenceSpec(n_evidence=2)),
        query_types=["clg", "discrete"],
        n_tasks=1,
        methods=["LW", "RBLW", "GS", "RBGS"],
        sample_sizes={"lw": [100], "gibbs": [100]},
        n_reps=1,
        reference=reference,
        n_reference=20_000,
        n_reference_gibbs=0,
        n_prior=500,
        gibbs_options={"n_chains": 10, "burn_in": 5},
    )


def test_run_unit_and_resume(tmp_path: Path) -> None:
    """Test that a unit writes finite errors for every method, and that a finished unit is skipped when resuming."""
    unit = _small_unit("smoke")
    n_rows = run_unit(unit, tmp_path / "smoke")
    results = pd.read_csv(tmp_path / "smoke" / "results.csv")
    assert n_rows == len(results) > 0
    assert set(results["method"]) == {"LW", "RBLW", "GS", "RBGS"}
    references = [json.loads(line) for line in (tmp_path / "smoke" / "references.jsonl").read_text().splitlines()]
    ok_tasks = {r["task_id"] for r in references if r["ok"]}
    errors = np.where(results["query_type"] == "discrete", results.get("prob_error"), results.get("err_mean"))
    assert np.all(np.isfinite(errors[results["task_id"].isin(ok_tasks)]))
    assert run_unit(unit, tmp_path / "smoke") == -1


def test_classifier_unit(tmp_path: Path) -> None:
    """Test the classification with missing features of an SPBN classifier on iris."""
    unit = UnitSpec(
        unit_id="iris_SP-NB",
        kind="classifier",
        factors={"dataset": "iris", "model": "SP-NB"},
        config={"dataset": "iris", "model": "SP-NB", "max_train_size": 200},
        seed=SEED,
        evidence={},
        query_types=["discrete"],
        n_tasks=2,
        methods=["LW", "RBLW", "GS", "RBGS"],
        sample_sizes={"lw": [200], "gibbs": [200]},
        n_reps=1,
        reference="stratified_lw",
        n_reference=5_000,
        gibbs_options={"n_chains": 10, "burn_in": 5},
        extra={"missing_rates": [0.5]},
    )
    run_unit(unit, tmp_path / "iris")
    results = pd.read_csv(tmp_path / "iris" / "results.csv")
    assert set(results["method"]) == {"LW", "RBLW", "GS", "RBGS", "MeanImpute", "kNNImpute"}
    assert np.all(np.isfinite(results["log_loss"])) and results["brier"].between(0, 2).all()


def test_analyze_comparison(tmp_path: Path) -> None:
    """Test the comparison tables on synthetic results where RBLW halves the error of LW at equal N and equal time."""
    rng = np.random.default_rng(SEED)
    rows = []
    for task in range(8):
        for n_samples in [100, 1000, 10000]:
            for method, scale in [("LW", 1.0), ("RBLW", 0.5)]:
                for rep in range(10):
                    rows.append({
                        "task_id": f"t{task}", "query_type": "clg", "method": method, "n_samples": n_samples,
                        "err_mean": scale * rng.normal() / np.sqrt(n_samples), "prob_error": np.nan,
                        "runtime": 1e-4 * n_samples * (1 + 0.01 * rng.random()), "w1": np.nan,
                    })
    results = pd.DataFrame(rows)
    results["error"] = results["err_mean"]
    analyze_comparison(results, tmp_path, 0.01, "test", "test")
    tests = pd.read_csv(tmp_path / "tables" / "test_paired_tests.csv")
    equal_n = tests[tests["comparison"] == "equal N"]
    assert np.all(equal_n["median_reduction"].between(0.3, 0.7))
    assert (tmp_path / "figures" / "test_rmse_vs_time.png").exists()
    summary = task_summary(results)
    assert set(summary.columns) >= {"rmse", "rmse_se", "runtime"}
