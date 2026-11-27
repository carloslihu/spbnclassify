"""Tests of the inference experiments (src/experiments/inference/run.py) and of their analysis (src/experiments/inference/analysis.py)."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pybnesian as pbn
import pytest
from scipy.stats import norm
from src.bn import SemiParametricBayesianNetwork
from src.experiments.inference.analysis import (
    analyze_comparison,
    holm_correction,
    task_summary,
    time_to_epsilon,
)
from src.experiments.inference.run import (
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
    references_agree,
    repair_reference_checks,
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
    config = SyntheticConfig(
        n_nodes=10, n_discrete=2, n_categories=4, n_discrete_parents_R=2, n_train=500
    )
    _, info = generate_synthetic_spbn(config, SEED)
    assert info["n_delta"] == 16
    # 64 configurations: the balanced design gives every configuration training data, so plain LW samples finite values
    config = SyntheticConfig(
        n_nodes=10, n_discrete=3, n_categories=4, n_discrete_parents_R=3, n_train=1000
    )
    bn, info = generate_synthetic_spbn(config, SEED)
    assert info["n_delta"] == 64
    samples = bn.sample(2000, seed=SEED)
    assert np.all(
        np.isfinite(samples[bn._rb_nodes(bn._sampled_nodes())].to_numpy(dtype=float))
    )


def test_task_evidence_spec() -> None:
    """Test the evidence location and the number of unobserved CLG ancestors of the evidence on R."""
    bn, _ = generate_synthetic_spbn(
        SyntheticConfig(n_nodes=20, chain_prob=1.0, n_train=300), SEED
    )
    prior = bn.sample(500, seed=SEED)
    sampled = bn._sampled_nodes()
    for location in ["S", "R"]:
        tasks = generate_tasks(
            bn, 3, ["clg"], EvidenceSpec(n_evidence=2, location=location), SEED, prior
        )
        for task in tasks:
            assert all((node in sampled) == (location == "S") for node in task.evidence)
            assert task.query not in task.evidence
    tasks = generate_tasks(
        bn,
        3,
        ["s_continuous"],
        EvidenceSpec(n_evidence=3, n_rb_ancestors=2),
        SEED,
        prior,
    )
    assert tasks and all(
        task.factors["n_unobserved_rb_ancestors"] == 2 for task in tasks
    )


def test_grid_reference_matches_gaussian() -> None:
    """Test that the grid reference of a Gaussian network matches its exact posterior."""
    bn, _ = generate_synthetic_spbn(
        SyntheticConfig(n_nodes=4, frac_clg=1.0, n_discrete=0, n_train=500), SEED
    )
    prior = bn.sample(2000, seed=SEED)
    tasks = generate_tasks(
        bn, 1, ["clg"], EvidenceSpec(n_evidence=1, location="R"), SEED, prior
    )
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
    tasks = generate_tasks(
        bn,
        1,
        ["clg", "s_continuous", "discrete"],
        EvidenceSpec(n_evidence=2),
        SEED,
        prior,
    )
    assert tasks
    for task in tasks:
        grid = grid_reference(bn, task, prior)
        lw = lw_reference(bn, task, 200_000, SEED)
        if task.query_type == "discrete":
            for category, probability in grid.probabilities.items():
                assert (
                    abs(probability - lw.probabilities[category]) < 5 * lw.mcse + 1e-3
                )
        else:
            assert abs(grid.mean - lw.mean) < 5 * lw.mcse + 1e-3


def test_discrete_reference_check() -> None:
    """Test that a discrete query is checked with stratified likelihood weighting, which agrees with the grid reference and
    has a finite Monte Carlo standard error."""
    config = SyntheticConfig(n_nodes=5, frac_clg=0.4, n_discrete=1, n_train=300)
    bn, _ = generate_synthetic_spbn(config, SEED)
    prior = bn.sample(2000, seed=SEED)
    task = generate_tasks(bn, 1, ["discrete"], EvidenceSpec(n_evidence=2), SEED, prior)[
        0
    ]
    reference = checked_reference(bn, task, 50_000, 20_000, SEED)
    assert reference.method == "lw+stratified" and reference.ok, reference.message
    stratified = stratified_class_reference(bn, task, 20_000, SEED)
    assert np.isfinite(stratified.mcse) and stratified.mcse > 0
    grid = grid_reference(bn, task, prior)
    for category, probability in grid.probabilities.items():
        assert (
            abs(stratified.probabilities[category] - probability)
            < 5 * stratified.mcse + 1e-3
        )


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
    np.testing.assert_allclose(
        mixture["std"] ** 2,
        np.sum(weights * (np.array([0.25, 1.0]) + (means - mean) ** 2)),
    )

    rng = np.random.default_rng(SEED)
    samples = rng.normal(size=500)
    uniform = np.full(500, 1 / 500)
    kde = kde_estimate(uniform, samples, grid)
    assert kde["bandwidth"] == SemiParametricBayesianNetwork._kde_bandwidth(
        samples, uniform
    )
    np.testing.assert_allclose(kde["std"] ** 2, samples.var() + kde["bandwidth"] ** 2)
    assert kde["cdf"][0] < 1e-6 and kde["cdf"][-1] > 1 - 1e-6


def test_holm_and_time_to_epsilon() -> None:
    """Test the Holm correction and the log-log interpolation of the time to reach a target error."""
    np.testing.assert_allclose(
        holm_correction(np.array([0.01, 0.04, 0.03, np.nan])),
        [0.03, 0.06, 0.06, np.nan],
    )
    curve = pd.DataFrame(
        {
            "runtime": [1.0, 10.0, 1000.0],
            "rmse": [1.0, 1 / np.sqrt(10), 1 / np.sqrt(1000)],
        }
    )
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
    references = [
        json.loads(line)
        for line in (tmp_path / "smoke" / "references.jsonl").read_text().splitlines()
    ]
    ok_tasks = {r["task_id"] for r in references if r["ok"]}
    errors = np.where(
        results["query_type"] == "discrete",
        results.get("prob_error"),
        results.get("err_mean"),
    )
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
    assert set(results["method"]) == {
        "LW",
        "RBLW",
        "GS",
        "RBGS",
        "MeanImpute",
        "kNNImpute",
    }
    assert (
        np.all(np.isfinite(results["log_loss"]))
        and results["brier"].between(0, 2).all()
    )


def test_analyze_comparison(tmp_path: Path) -> None:
    """Test the comparison tables on synthetic results where RBLW halves the error of LW at equal N and equal time."""
    rng = np.random.default_rng(SEED)
    rows = []
    for task in range(8):
        for n_samples in [100, 1000, 10000]:
            for method, scale in [("LW", 1.0), ("RBLW", 0.5)]:
                for rep in range(10):
                    rows.append(
                        {
                            "task_id": f"t{task}",
                            "query_type": "clg",
                            "method": method,
                            "n_samples": n_samples,
                            "err_mean": scale * rng.normal() / np.sqrt(n_samples),
                            "prob_error": np.nan,
                            "runtime": 1e-4 * n_samples * (1 + 0.01 * rng.random()),
                            "w1": np.nan,
                        }
                    )
    results = pd.DataFrame(rows)
    results["error"] = results["err_mean"]
    analyze_comparison(results, tmp_path, 0.01, "test", "test")
    tests = pd.read_csv(tmp_path / "tables" / "test_paired_tests.csv")
    equal_n = tests[tests["comparison"] == "equal N"]
    assert np.all(equal_n["median_reduction"].between(0.3, 0.7))
    assert (tmp_path / "figures" / "test_rmse_vs_time.png").exists()
    summary = task_summary(results)
    assert set(summary.columns) >= {"rmse", "rmse_se", "runtime"}


def test_reference_check_table(tmp_path: Path) -> None:
    """Test the fraction of discarded tasks and the reasons of the reference checks."""
    from src.experiments.inference.analysis import reference_check_table

    references = [
        {
            "task_id": "frac_clg=0.3_0/clg_0",
            "method": "lw+gibbs",
            "ok": True,
            "message": "",
        },
        {
            "task_id": "frac_clg=0.3_1/clg_0",
            "method": "lw+gibbs",
            "ok": False,
            "message": "LW reference ESS 12 < 1000",
        },
        {
            "task_id": "frac_clg=0.3_1/s_continuous_0",
            "method": "lw+gibbs",
            "ok": False,
            "message": "Gibbs check R-hat 1.2 >= 1.01",
        },
        {
            "task_id": "frac_clg=0.5_0/discrete_0",
            "method": "lw+stratified",
            "ok": False,
            "message": "LW and stratified_lw references differ by 0.1 > 0.01",
        },
    ]
    tasks = [
        {
            "task_id": r["task_id"],
            "query_type": r["task_id"].split("/")[1].rsplit("_", 1)[0],
        }
        for r in references
    ]
    for name, rows in [("references.jsonl", references), ("tasks.jsonl", tasks)]:
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    table = reference_check_table(tmp_path, "test").set_index(
        ["configuration", "query_type"]
    )
    assert table.loc[("frac_clg=0.3", "clg"), "frac_discarded"] == 0.5
    assert table.loc[
        ("all", "all"), ["n_low_ess", "n_not_converged", "n_disagreement", "n_kept"]
    ].tolist() == [1, 1, 1, 1]
    assert (tmp_path / "tables" / "test_reference_check.tex").exists()


def test_reference_and_run_stages(tmp_path: Path) -> None:
    """Test that the timed stage reuses the network, tasks and references of the reference stage, and that it gives the
    same estimates as running both stages at once (the pickled network is the same network).
    """
    from src.experiments.inference.run import prepare_unit

    unit = _small_unit("staged")
    n_tasks = prepare_unit(unit, tmp_path / "staged")
    assert n_tasks > 0 and (tmp_path / "staged" / "references_done").exists()
    assert (tmp_path / "staged" / "network.pkl").exists() and not (
        tmp_path / "staged" / "done"
    ).exists()
    assert prepare_unit(unit, tmp_path / "staged") == -1
    references_time = (tmp_path / "staged" / "references.jsonl").stat().st_mtime
    run_unit(unit, tmp_path / "staged")
    assert (tmp_path / "staged" / "references.jsonl").stat().st_mtime == references_time

    run_unit(_small_unit("staged"), tmp_path / "all_at_once")
    staged = pd.read_csv(tmp_path / "staged" / "results.csv")
    all_at_once = pd.read_csv(tmp_path / "all_at_once" / "results.csv")
    columns = [
        c for c in ["mean", "std", "w1", "probabilities", "n_eff"] if c in staged
    ]
    pd.testing.assert_frame_equal(staged[columns], all_at_once[columns])


def test_worker_slots() -> None:
    """Test the round-robin assignment of GPUs and cores to the workers, and that a worker takes its slot."""
    import os
    import queue

    from src.utils.opencl_device import assign_worker_device, worker_slots

    slots = worker_slots(3, ["g0", "g1"], [4, 5, 6, 7])
    assert slots == [("g0", 4), ("g1", 5), ("g0", 6)]
    assert worker_slots(2, [], None) == [(None, None), (None, None)]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    affinity = os.sched_getaffinity(0)
    core = min(affinity)
    slot_queue = queue.Queue()
    slot_queue.put(("test-gpu", core))
    try:
        assign_worker_device(slot_queue)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "test-gpu"
        assert os.sched_getaffinity(0) == {core}
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        os.sched_setaffinity(0, affinity)


def test_pair_tests_per_unit() -> None:
    """Test that the paired tests use one value per network (the median of its tasks), not one per task."""
    from src.experiments.inference.analysis import pair_tests

    rows = []
    for network in range(6):
        for task in range(3):
            for method, rmse in [("LW", 1.0 + 0.1 * task), ("RBLW", 0.5 + 0.1 * task)]:
                rows.append(
                    {
                        "task_id": f"network_{network}/clg_{task}",
                        "query_type": "clg",
                        "method": method,
                        "n_samples": 100,
                        "rmse": rmse,
                        "runtime": 1.0,
                    }
                )
    tests = pair_tests(pd.DataFrame(rows), {})
    row = tests[
        (tests["pair"] == "RBLW vs LW") & (tests["comparison"] == "equal N")
    ].iloc[0]
    assert row["n_units"] == 6
    np.testing.assert_allclose(row["median_reduction"], (1.1 - 0.6) / 1.1)
    assert row["frac_better"] == 1.0 and row["p_value"] < 0.05


def test_regime_base_levels() -> None:
    """Test that delta = 1 is the base configuration, that the E3 base units are shared by the factors (no duplicated
    units), and that the base levels of the runner and of the analysis agree."""
    from src.experiments.inference import analysis as inference_analysis
    from src.experiments.inference import run as inference_experiments
    from src.experiments.inference.run import (
        _network_factor_config,
        build_units,
        parse_args,
    )

    assert _network_factor_config("n_discrete_parents_R", 0) == asdict(
        SyntheticConfig()
    )
    assert (
        inference_analysis.REGIME_BASE_LEVELS
        == inference_experiments.REGIME_BASE_LEVELS
    )
    assert (
        inference_analysis.NEAR_CERTAIN_PROBABILITY
        == inference_experiments.NEAR_CERTAIN_PROBABILITY
    )
    units = build_units("E3", parse_args(["E3"]))
    signatures = [
        (
            json.dumps(u.config, sort_keys=True),
            u.seed,
            json.dumps(u.evidence, sort_keys=True),
        )
        for u in units
    ]
    assert len(signatures) == len(set(signatures))
    base = [u for u in units if u.factors["factor"] == "base"]
    assert len(base) == 10 and all(u.config == asdict(SyntheticConfig()) for u in base)


def test_lw_reference_extends_for_rare_evidence() -> None:
    """Test that the LW reference keeps sampling while its ESS is low, and that not storing negligible samples does not
    change it."""
    import src.experiments.inference.run as experiments

    bn, _ = generate_synthetic_spbn(SyntheticConfig(n_nodes=8, n_train=300), SEED)
    prior = bn.sample(2000, seed=SEED)
    tasks = generate_tasks(
        bn, 1, ["clg"], EvidenceSpec(n_evidence=4, rarity=0.001), SEED, prior
    )
    task = tasks[0]
    fixed = lw_reference(bn, task, 1000, SEED)
    assert (
        fixed.n_samples == 1000
        and fixed.effective_sample_size < experiments.MIN_REFERENCE_ESS
    )
    extended = lw_reference(bn, task, 1000, SEED, max_samples=300_000)
    assert extended.n_samples > 1000
    assert (
        extended.effective_sample_size >= experiments.MIN_REFERENCE_ESS
        or extended.n_samples == 300_000
    )
    assert extended.effective_sample_size > fixed.effective_sample_size

    original = experiments.REFERENCE_PRUNE_LOG_WEIGHT
    try:
        experiments.REFERENCE_PRUNE_LOG_WEIGHT = np.inf
        unpruned = lw_reference(bn, task, 1000, SEED, max_samples=300_000)
    finally:
        experiments.REFERENCE_PRUNE_LOG_WEIGHT = original
    assert unpruned.n_samples == extended.n_samples
    np.testing.assert_allclose(extended.mean, unpruned.mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        extended.effective_sample_size, unpruned.effective_sample_size, rtol=1e-9
    )


def test_primary_endpoint_and_near_certain(tmp_path: Path) -> None:
    """Test that the paired tests mark one primary test per pair and query type (equal time at the largest budget) with
    its own Holm correction, and that near-certain discrete tasks are tested apart."""
    from src.experiments.inference.analysis import N_BUDGETS, without_near_certain

    rng = np.random.default_rng(SEED)
    rows = []
    for network in range(6):
        for query_type, max_probability in [("clg", np.nan), ("discrete", 0.999)]:
            for n_samples in [100, 1000, 10000]:
                for method, scale in [("LW", 1.0), ("RBLW", 0.5)]:
                    for rep in range(5):
                        rows.append(
                            {
                                "task_id": f"network_{network}/{query_type}_0",
                                "query_type": query_type,
                                "method": method,
                                "n_samples": n_samples,
                                "reference_max_probability": max_probability,
                                "err_mean": (
                                    scale * rng.normal() / np.sqrt(n_samples)
                                    if query_type == "clg"
                                    else np.nan
                                ),
                                "prob_error": (
                                    scale * abs(rng.normal()) / np.sqrt(n_samples)
                                    if query_type == "discrete"
                                    else np.nan
                                ),
                                "runtime": 1e-4 * n_samples * (1 + 0.01 * rng.random()),
                            }
                        )
    results = pd.DataFrame(rows)
    results["error"] = np.where(
        results["query_type"] == "discrete", results["prob_error"], results["err_mean"]
    )
    analyze_comparison(results, tmp_path, 0.01, "test", "test")
    tests = pd.read_csv(tmp_path / "tables" / "test_paired_tests.csv")
    primary = tests[tests["endpoint"] == "primary"]
    assert (
        len(primary) == 2
        and primary["level"].str.startswith(f"budget {N_BUDGETS}/").all()
    )
    np.testing.assert_allclose(
        primary["p_holm"], holm_correction(primary["p_value"].to_numpy())
    )
    without = pd.read_csv(
        tmp_path / "tables" / "test_paired_tests_without_near_certain.csv"
    )
    assert set(without["query_type"]) == {"clg"}
    summary = task_summary(results)
    assert set(without_near_certain(summary)["query_type"]) == {"clg"}


def test_e3_shared_base(tmp_path: Path) -> None:
    """Test that the E3 analysis uses the shared base units as the base level of every factor."""
    from src.experiments.inference.analysis import analyze_e3

    rng = np.random.default_rng(SEED)
    rows = []
    for factor, level in [("base", "base"), ("frac_clg", 0.9), ("location", "R")]:
        for network in range(6):
            for n_samples in [100, 1000]:
                for method, scale in [
                    ("LW", 1.0),
                    ("RBLW", 0.5),
                    ("GS", 1.0),
                    ("RBGS", 0.5),
                ]:
                    for rep in range(3):
                        rows.append(
                            {
                                "task_id": f"{factor}={level}_{network}/clg_0",
                                "query_type": "clg",
                                "method": method,
                                "n_samples": n_samples,
                                "factor": factor,
                                "level": level,
                                "n_eff": 50.0,
                                "err_mean": scale * rng.normal(),
                                "runtime": 1e-4 * n_samples,
                            }
                        )
    results = pd.DataFrame(rows)
    results["error"] = results["err_mean"]
    analyze_e3(results, tmp_path)
    table = pd.read_csv(tmp_path / "tables" / "E3_regimes.csv")
    levels = set(zip(table["factor"], table["level"].astype(str)))
    assert {
        ("frac_clg", "0.5"),
        ("frac_clg", "0.9"),
        ("location", "both"),
        ("location", "R"),
        ("rarity", "0.5"),
        ("n_discrete_parents_R", "0"),
    } <= levels
    assert "base" not in set(table["factor"])


def test_reference_stage_skips_finished_units(tmp_path: Path) -> None:
    """Test that the reference stage does not recompute the references of a finished unit (e.g., one run before the stages
    were split, which has no references_done marker), so its results stay paired with their references.
    """
    from src.experiments.inference.run import prepare_unit

    unit_path = tmp_path / "finished"
    unit_path.mkdir()
    (unit_path / "references.jsonl").write_text("old\n")
    (unit_path / "done").touch()
    assert prepare_unit(_small_unit("finished"), unit_path) == -1
    assert (unit_path / "references.jsonl").read_text() == "old\n"
    assert not (unit_path / "references_done").exists()


def test_references_agree_with_zero_tolerance() -> None:
    """Identical near-certain estimates (MCSE 0, so tolerance 0 or NaN) agree; real differences still fail."""
    assert references_agree(1e-14, float("nan"))
    assert references_agree(6e-203, 0.0)
    assert references_agree(0.01, 0.02)
    assert not references_agree(0.03, 0.02)
    assert not references_agree(0.01, float("nan"))


def test_repair_reference_checks(tmp_path: Path) -> None:
    """Only the checks failed by identical estimates are repaired; real disagreements and other failures stay."""
    rows = [
        {
            "task_id": "a",
            "ok": False,
            "message": "LW and stratified_lw references differ by 1.377e-14 > nan",
        },
        {
            "task_id": "b",
            "ok": False,
            "message": "LW and stratified_lw references differ by 6.249e-203 > 0",
        },
        {
            "task_id": "c",
            "ok": False,
            "message": "LW and gibbs references differ by 0.05 > 0.01",
        },
        {"task_id": "d", "ok": False, "message": "LW reference ESS 12 < 1000"},
        {"task_id": "e", "ok": True, "message": ""},
    ]
    with open(tmp_path / "references.jsonl", "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    assert repair_reference_checks([tmp_path, tmp_path / "missing"]) == 2
    with open(tmp_path / "references.jsonl") as f:
        ok = {row["task_id"]: row["ok"] for row in map(json.loads, f)}
    assert ok == {"a": True, "b": True, "c": False, "d": False, "e": True}
    assert repair_reference_checks([tmp_path]) == 0


def test_run_unit_split_by_device(tmp_path: Path) -> None:
    """Test that the GPU and CPU parts of the timed stage give the same results as running every method at once."""
    unit = _small_unit("split")
    assert run_unit(unit, tmp_path / "split", "cpu") > 0
    path = tmp_path / "split"
    assert (path / "done_cpu").exists() and not (path / "done").exists()
    assert run_unit(unit, path, "cpu") == -1
    assert run_unit(unit, path, "gpu") > 0
    assert (path / "done").exists()

    run_unit(_small_unit("split"), tmp_path / "together")
    columns = ["task_id", "method", "n_samples", "seed", "mean", "std", "n_eff"]
    split = pd.read_csv(path / "results.csv")
    together = pd.read_csv(tmp_path / "together" / "results.csv")
    split, together = (
        df[[c for c in columns if c in df]]
        .sort_values(["task_id", "method", "n_samples", "seed"])
        .reset_index(drop=True)
        for df in (split, together)
    )
    pd.testing.assert_frame_equal(split, together)
    network = json.loads((path / "network.json").read_text())
    assert "gpu_runs_device" in network and "cpu_runs_cpu_cores" in network


def test_run_unit_cpu_shards(tmp_path: Path) -> None:
    """Test that the CPU part split in shards of tasks gives the same results as running it at once."""
    unit = _small_unit("shards")
    path = tmp_path / "shards"
    assert run_unit(unit, path, "cpu", (0, 2)) > 0
    assert (path / "done_cpu_0of2").exists() and not (path / "done_cpu").exists()
    assert run_unit(unit, path, "cpu", (0, 2)) == -1
    run_unit(unit, path, "cpu", (1, 2))
    assert (path / "done_cpu").exists() and not (path / "done").exists()
    run_unit(unit, path, "gpu")
    assert (path / "done").exists()

    run_unit(_small_unit("shards"), tmp_path / "together")
    columns = ["task_id", "method", "n_samples", "seed", "mean", "std", "n_eff"]
    sharded, together = (
        df[[c for c in columns if c in df]]
        .sort_values(["task_id", "method", "n_samples", "seed"])
        .reset_index(drop=True)
        for df in (
            pd.read_csv(path / "results.csv"),
            pd.read_csv(tmp_path / "together" / "results.csv"),
        )
    )
    pd.testing.assert_frame_equal(sharded, together)


def test_run_method_gibbs_checkpoints() -> None:
    """Test that a Gibbs method run once for several sample sizes gives the same rows as separate runs."""
    from src.experiments.inference.run import GIBBS_DEFAULTS, ckde_sizes, run_method

    config = SyntheticConfig(n_nodes=8, n_train=200)
    bn, _ = generate_synthetic_spbn(config, SEED)
    prior = bn.sample(500, seed=SEED)
    task = generate_tasks(bn, 1, ["clg"], EvidenceSpec(n_evidence=2), SEED, prior)[0]
    reference = lw_reference(bn, task, 20_000, SEED)
    options = {**GIBBS_DEFAULTS, "n_chains": 10, "burn_in": 5}
    sizes = ckde_sizes(bn)
    rows = run_method(bn, task, "RBGS", [50, 200], 0, reference, sizes, options)
    assert [row["n_samples"] for row in rows] == [50, 200]
    assert rows[0]["runtime"] <= rows[1]["runtime"]
    assert rows[0]["kernel_evaluations"] <= rows[1]["kernel_evaluations"]
    for row in rows:
        separate = run_method(
            bn, task, "RBGS", row["n_samples"], 0, reference, sizes, options
        )
        for key in ["mean", "std", "w1", "n_eff", "rhat", "mean_exact_rate"]:
            assert row[key] == pytest.approx(separate[key], nan_ok=True)
        assert row["kernel_evaluations"] == separate["kernel_evaluations"]
