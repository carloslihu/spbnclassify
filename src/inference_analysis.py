"""Figures, tables and statistical tests of the inference experiments (``inference_experiments.py``).

Reads ``data/results/inference/<experiment>/results.csv`` and writes ``figures/`` (PNG/PDF) and ``tables/`` (CSV and
booktabs LaTeX) in the same folder.

Usage (from the library root):
    python -m src.inference_analysis E2 [--quick] [--epsilon 0.1]
"""

# region Imports
import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .utils.model_comparison import (
    compute_friedman_test,
    get_ranking_matrix,
    plot_critical_difference_diagram,
)

# endregion Imports

LIBRARY_ROOT_PATH = Path(__file__).resolve().parent.parent
RESULT_PATH = LIBRARY_ROOT_PATH / "data" / "results" / "inference"
# Each Rao-Blackwellized method and its non-Rao-Blackwellized counterpart (hypotheses H1 and H2)
PAIRS = [("LW", "RBLW"), ("GS", "RBGS")]
METHOD_COLORS = {"LW": "#1976D2", "RBLW": "#FFB300", "GS": "#388E3C", "RBGS": "#D81B60"}
N_BOOTSTRAP = 2000
N_BUDGETS = 12
# Errors below this tolerance are exact up to the precision of the reference (the grid integration error is ~1e-7)
EXACT_TOLERANCE = 1e-5


# region Error aggregation
def add_error_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Adds ``error``, the scalar error of every run: the normalized mean error for continuous queries and the maximum
    absolute probability error for discrete queries (both are 0 for a perfect estimate)
    """
    results = results.copy()
    if "err_mean" not in results:
        results["err_mean"] = np.nan
    if "prob_error" not in results:
        results["prob_error"] = np.nan
    if "query_type" not in results:
        results["query_type"] = "discrete"
    results["error"] = np.where(
        results["query_type"] == "discrete", results["prob_error"], results["err_mean"]
    )
    return results


def task_summary(
    results: pd.DataFrame, extra_keys: list[str] | None = None
) -> pd.DataFrame:
    """RMSE over the repetitions of every (task, method, sample size), with its Monte Carlo standard error (delta method),
    and the mean runtime, kernel evaluations, W1 distance and diagnostics"""
    keys = ["task_id", "query_type", "method", "n_samples"] + (extra_keys or [])
    keys = [k for k in keys if k in results]
    aggregations = {
        "squared_error": ("error", lambda e: np.mean(np.square(e))),
        "squared_error_se": (
            "error",
            lambda e: (
                np.std(np.square(e), ddof=1) / np.sqrt(len(e)) if len(e) > 1 else np.nan
            ),
        ),
        "runtime": ("runtime", "mean"),
        "n_reps": ("error", "size"),
    }
    for column in [
        "w1",
        "err_std",
        "kernel_evaluations",
        "n_eff",
        "rhat",
        "max_rhat",
        "mean_exact_rate",
        "mean_acceptance_rate",
        "mean_block_acceptance_rate",
        "log_score",
    ]:
        if column in results:
            aggregations[column] = (column, "mean")
    summary = results.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    summary["rmse"] = np.sqrt(summary["squared_error"])
    summary["rmse_se"] = summary["squared_error_se"] / (
        2 * summary["rmse"].replace(0, np.nan)
    )
    return summary.drop(columns=["squared_error", "squared_error_se"])


def _loglog_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Interpolation of y(x) in log-log scale, NaN outside the range of x"""
    order = np.argsort(x)
    x, y = np.asarray(x, dtype=float)[order], np.asarray(y, dtype=float)[order]
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return np.full(len(x_new), np.nan)
    return np.exp(
        np.interp(np.log(x_new), np.log(x), np.log(y), left=np.nan, right=np.nan)
    )


def time_to_epsilon(curve: pd.DataFrame, epsilon: float) -> float:
    """Time t_epsilon at which the RMSE(t) curve of a task and method (rows over the sample sizes) first reaches epsilon,
    interpolated in log-log scale. It is 0 if the cheapest run already reaches it and infinity if no run reaches it.
    """
    curve = curve.sort_values("runtime")
    times, errors = curve["runtime"].to_numpy(), curve["rmse"].to_numpy()
    if errors[0] <= epsilon:
        return 0.0
    for i in range(1, len(errors)):
        if errors[i] <= epsilon:
            if errors[i - 1] <= 0 or errors[i] <= 0:
                return float(times[i])
            fraction = (np.log(errors[i - 1]) - np.log(epsilon)) / (
                np.log(errors[i - 1]) - np.log(errors[i])
            )
            return float(
                np.exp(
                    np.log(times[i - 1])
                    + fraction * (np.log(times[i]) - np.log(times[i - 1]))
                )
            )
    return np.inf


def rmse_at_budgets(
    summary: pd.DataFrame, methods: tuple[str, str], n_budgets: int = N_BUDGETS
) -> pd.DataFrame:
    """RMSE of both methods of a pair at common wall-clock budgets per task: geometric budgets over the range of times
    that both methods cover, interpolated in log-log scale"""
    rows = []
    for task_id, task in summary[summary["method"].isin(methods)].groupby("task_id"):
        curves = {m: task[task["method"] == m] for m in methods}
        if any(len(c) < 2 for c in curves.values()):
            continue
        lo = max(c["runtime"].min() for c in curves.values())
        hi = min(c["runtime"].max() for c in curves.values())
        if not hi > lo:
            continue
        budgets = np.geomspace(lo, hi, n_budgets)
        for budget_index, budget in enumerate(budgets):
            row = {
                "task_id": task_id,
                "query_type": task["query_type"].iloc[0],
                "budget_index": budget_index,
                "budget": budget,
            }
            for m, c in curves.items():
                row[m] = _loglog_interpolate(
                    c["runtime"].to_numpy(), c["rmse"].to_numpy(), np.array([budget])
                )[0]
            rows.append(row)
    return pd.DataFrame(rows)


# endregion Error aggregation


# region Statistical tests
def holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN p-values are left as NaN and do not count as tests)"""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    order = valid[np.argsort(p_values[valid])]
    m = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def bootstrap_ci(
    values: np.ndarray, statistic=np.median, n: int = N_BOOTSTRAP, seed: int = 0
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = [statistic(rng.choice(values, len(values))) for _ in range(n)]
    return tuple(np.quantile(samples, [0.025, 0.975]))


def paired_comparison(baseline: np.ndarray, proposed: np.ndarray) -> dict:
    """Paired comparison of per-task RMSEs: the median relative reduction (baseline - proposed) / baseline with a bootstrap
    95% CI, the fraction of tasks where the proposed method is better, and the Wilcoxon signed-rank p-value
    """
    baseline, proposed = np.asarray(baseline, dtype=float), np.asarray(
        proposed, dtype=float
    )
    valid = np.isfinite(baseline) & np.isfinite(proposed) & (baseline > 0)
    baseline, proposed = baseline[valid], proposed[valid]
    reduction = (baseline - proposed) / baseline
    p_value = np.nan
    if len(reduction) >= 5 and np.any(baseline != proposed):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_value = float(wilcoxon(baseline, proposed, zero_method="wilcox").pvalue)
    lo, hi = bootstrap_ci(reduction)
    return {
        "n_tasks": int(len(reduction)),
        "median_reduction": float(np.median(reduction)) if len(reduction) else np.nan,
        "ci_low": lo,
        "ci_high": hi,
        "frac_better": (
            float(np.mean(proposed < baseline)) if len(reduction) else np.nan
        ),
        "p_value": p_value,
    }


def pair_tests(
    summary: pd.DataFrame, budgets: dict[tuple, pd.DataFrame]
) -> pd.DataFrame:
    """Paired comparisons of every pair, query type and sample size (equal N) or budget (equal time), Holm-corrected
    together"""
    rows = []
    for baseline, proposed in PAIRS:
        for query_type, by_type in summary.groupby("query_type"):
            wide = by_type.pivot_table(
                index=["task_id", "n_samples"], columns="method", values="rmse"
            )
            if baseline not in wide or proposed not in wide:
                continue
            for n_samples, group in wide.reset_index().groupby("n_samples"):
                rows.append(
                    {
                        "pair": f"{proposed} vs {baseline}",
                        "query_type": query_type,
                        "comparison": "equal N",
                        "level": n_samples,
                        **paired_comparison(group[baseline], group[proposed]),
                    }
                )
            at_budgets = budgets.get((baseline, proposed), pd.DataFrame())
            if len(at_budgets):
                for budget_index, group in at_budgets[
                    at_budgets["query_type"] == query_type
                ].groupby("budget_index"):
                    rows.append(
                        {
                            "pair": f"{proposed} vs {baseline}",
                            "query_type": query_type,
                            "comparison": "equal time",
                            "level": f"budget {budget_index + 1}/{N_BUDGETS} (median {group['budget'].median():.3g} s)",
                            **paired_comparison(group[baseline], group[proposed]),
                        }
                    )
    tests = pd.DataFrame(rows)
    if len(tests):
        # Sample sizes that only one family uses (e.g., the Gibbs sizes for the LW pair) leave no paired tasks
        tests = tests[tests["n_tasks"] > 0].reset_index(drop=True)
    if len(tests):
        tests["p_holm"] = holm_correction(tests["p_value"].to_numpy())
    return tests


# endregion Statistical tests


# region Output helpers
def save_table(
    df: pd.DataFrame, path: Path, caption: str = "", float_format: str = "%.3g"
) -> None:
    """Writes a table as CSV and as a booktabs LaTeX table"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        latex = df.to_latex(
            index=False,
            float_format=float_format,
            na_rep="--",
            caption=caption or None,
            label=f"tab:{path.stem}",
            escape=True,
        )
    path.with_suffix(".tex").write_text(latex)


def save_figure(fig: matplotlib.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_error_curves(summary: pd.DataFrame, x: str, path: Path, title: str) -> None:
    """Median over tasks (with the interquartile range) of the RMSE versus ``x`` ("n_samples" or "runtime") per query type
    and method. For runtime, the curves of each task are interpolated on common budgets before taking the median.
    """
    query_types = sorted(summary["query_type"].unique())
    fig, axes = plt.subplots(
        1, len(query_types), figsize=(4.5 * len(query_types), 3.6), squeeze=False
    )
    for ax, query_type in zip(axes[0], query_types):
        by_type = summary[summary["query_type"] == query_type]
        for method, group in by_type.groupby("method"):
            if x == "n_samples":
                stats = (
                    group.groupby("n_samples")["rmse"]
                    .quantile([0.25, 0.5, 0.75])
                    .unstack()
                )
                xs = stats.index.to_numpy(dtype=float)
            else:
                lo, hi = group["runtime"].min(), group["runtime"].max()
                if not hi > lo:
                    continue
                xs = np.geomspace(lo, hi, 30)
                curves = np.array(
                    [
                        _loglog_interpolate(
                            task["runtime"].to_numpy(), task["rmse"].to_numpy(), xs
                        )
                        for _, task in group.groupby("task_id")
                    ]
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    stats = pd.DataFrame(
                        np.nanquantile(curves, [0.25, 0.5, 0.75], axis=0).T,
                        columns=[0.25, 0.5, 0.75],
                    )
            color = METHOD_COLORS.get(method)
            ax.plot(
                xs,
                stats[0.5],
                marker="o" if x == "n_samples" else None,
                label=method,
                color=color,
            )
            ax.fill_between(xs, stats[0.25], stats[0.75], alpha=0.2, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(
            "Number of samples $N$" if x == "n_samples" else "Wall-clock time (s)"
        )
        ax.set_ylabel("RMSE")
        ax.set_title(f"{query_type} query")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, path)


def load_results(experiment_path: Path) -> pd.DataFrame:
    results_file = experiment_path / "results.csv"
    if not results_file.exists():
        raise FileNotFoundError(
            f"No results in {results_file}; run inference_experiments first."
        )
    return add_error_columns(pd.read_csv(results_file))


# endregion Output helpers


# region Experiment analyses
def analyze_comparison(
    results: pd.DataFrame, output: Path, epsilon: float, prefix: str, title: str
) -> pd.DataFrame:
    """E2 (and every comparison of the pairs): error curves, tests at equal N and equal time, and t_epsilon ratios"""
    summary = task_summary(results)
    save_table(
        summary, output / "tables" / f"{prefix}_task_summary", float_format="%.4g"
    )
    plot_error_curves(
        summary, "n_samples", output / "figures" / f"{prefix}_rmse_vs_n", title
    )
    plot_error_curves(
        summary, "runtime", output / "figures" / f"{prefix}_rmse_vs_time", title
    )
    budgets = {pair: rmse_at_budgets(summary, pair) for pair in PAIRS}
    tests = pair_tests(summary, budgets)
    if len(tests):
        save_table(
            tests,
            output / "tables" / f"{prefix}_paired_tests",
            caption="Median relative RMSE reduction of the Rao--Blackwellized methods, with bootstrap 95\\% CIs and "
            "Holm-corrected Wilcoxon signed-rank p-values.",
        )
    rows = []
    for baseline, proposed in PAIRS:
        for query_type, by_type in summary.groupby("query_type"):
            ratios = []
            for task_id, task in by_type.groupby("task_id"):
                curves = {m: task[task["method"] == m] for m in (baseline, proposed)}
                if any(len(c) == 0 for c in curves.values()):
                    continue
                t_base, t_prop = (
                    time_to_epsilon(curves[m], epsilon) for m in (baseline, proposed)
                )
                ratios.append(
                    t_prop / t_base if np.isfinite(t_base) and t_base > 0 else np.nan
                )
            ratios = np.asarray(ratios, dtype=float)
            finite = ratios[np.isfinite(ratios)]
            if len(ratios):
                rows.append(
                    {
                        "pair": f"{proposed} vs {baseline}",
                        "query_type": query_type,
                        "epsilon": epsilon,
                        "n_tasks": len(ratios),
                        "n_reached": len(finite),
                        "median_time_ratio": (
                            float(np.median(finite)) if len(finite) else np.nan
                        ),
                        "frac_faster": (
                            float(np.mean(finite < 1)) if len(finite) else np.nan
                        ),
                    }
                )
    if rows:
        save_table(
            pd.DataFrame(rows),
            output / "tables" / f"{prefix}_time_to_epsilon",
            caption=f"Ratio of the times needed to reach RMSE $\\leq {epsilon}$ (below 1 favors Rao--Blackwellization).",
        )
    diagnostics_table(summary, output, prefix)
    return summary


def diagnostics_table(summary: pd.DataFrame, output: Path, prefix: str) -> pd.DataFrame:
    """Diagnostics of every method, query type and sample size (medians over the tasks): the normalized effective sample
    size n_eff / N (importance sampling ESS for LW and RBLW, bulk ESS of the query for GS and RBGS), the ESS per second,
    and for the Gibbs samplers the R-hat of the query, the maximum R-hat and the exact, acceptance and block acceptance
    rates averaged over the variables"""
    if "n_eff" not in summary:
        return pd.DataFrame()
    summary = summary.copy()
    summary["neff_frac"] = summary["n_eff"] / summary["n_samples"]
    summary["ess_per_second"] = summary["n_eff"] / summary["runtime"]
    columns = [
        c
        for c in [
            "neff_frac",
            "ess_per_second",
            "rhat",
            "max_rhat",
            "mean_exact_rate",
            "mean_acceptance_rate",
            "mean_block_acceptance_rate",
        ]
        if c in summary and summary[c].notna().any()
    ]
    table = (
        summary.groupby(["method", "query_type", "n_samples"])[columns]
        .median()
        .reset_index()
    )
    save_table(
        table,
        output / "tables" / f"{prefix}_diagnostics",
        caption="Diagnostics of the inference methods (medians over the tasks): $n_{\\mathrm{eff}} / N$, ESS per second, "
        "$\\widehat{R}$ and the rates of the Gibbs samplers.",
    )
    return table


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def reference_check_table(output: Path, prefix: str) -> pd.DataFrame:
    """Fraction of the tasks discarded because their reference failed the checks of ``checked_reference`` (low ESS of the
    LW reference, non-converged Gibbs chains, or disagreement between both estimates), per configuration and query type.
    The configuration is the unit identifier without its network index (e.g., "frac_clg=0.3")."""
    references = _read_jsonl(output / "references.jsonl")
    tasks = _read_jsonl(output / "tasks.jsonl")
    if not len(references) or (references["method"] == "none").all():
        return pd.DataFrame()
    references = references.merge(tasks[["task_id", "query_type"]], on="task_id", how="left")
    unit = references["task_id"].str.split("/").str[0]
    references["configuration"] = unit.str.replace(r"_\d+$", "", regex=True)
    message = references["message"].fillna("")
    references["low_ess"] = message.str.contains("ESS")
    references["not_converged"] = message.str.contains("R-hat")
    references["disagreement"] = message.str.contains("differ")
    references["failed"] = ~references["ok"] & ~(
        references["low_ess"] | references["not_converged"] | references["disagreement"]
    )
    table = (
        references.groupby(["configuration", "query_type"])
        .agg(
            n_tasks=("ok", "size"),
            n_kept=("ok", "sum"),
            n_low_ess=("low_ess", "sum"),
            n_not_converged=("not_converged", "sum"),
            n_disagreement=("disagreement", "sum"),
            n_failed=("failed", "sum"),
        )
        .reset_index()
    )
    table["frac_discarded"] = 1.0 - table["n_kept"] / table["n_tasks"]
    total = table[["n_tasks", "n_kept", "n_low_ess", "n_not_converged", "n_disagreement", "n_failed"]].sum()
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [{"configuration": "all", "query_type": "all", **total.to_dict(),
                  "frac_discarded": 1.0 - total["n_kept"] / total["n_tasks"]}]
            ),
        ],
        ignore_index=True,
    )
    save_table(
        table,
        output / "tables" / f"{prefix}_reference_check",
        caption="Tasks discarded because their reference failed the checks: low effective sample size of the LW reference, "
        "non-converged Gibbs chains, disagreement between both estimates, or a numerical failure.",
    )
    return table


def analyze_e1(results: pd.DataFrame, output: Path) -> None:
    """E1: errors per check and the convergence slope of log RMSE versus log N (close to -1/2 for consistent estimators)"""
    summary = task_summary(results, ["check"])
    table = (
        summary.groupby(["check", "query_type", "method", "n_samples"])[["rmse", "w1"]]
        .median()
        .reset_index()
    )
    save_table(
        table,
        output / "tables" / "E1_errors",
        caption="Median RMSE and $W_1$ distance of the correctness checks.",
    )
    slopes = []
    for (check, query_type, method), group in summary.groupby(
        ["check", "query_type", "method"]
    ):
        task_slopes = []
        for _, task in group.groupby("task_id"):
            valid = task["rmse"] > EXACT_TOLERANCE
            if valid.sum() >= 2:
                task_slopes.append(
                    np.polyfit(
                        np.log(task.loc[valid, "n_samples"]),
                        np.log(task.loc[valid, "rmse"]),
                        1,
                    )[0]
                )
        exact = bool((group["rmse"] <= EXACT_TOLERANCE).all())
        slopes.append(
            {
                "check": check,
                "query_type": query_type,
                "method": method,
                "slope": float(np.median(task_slopes)) if task_slopes else np.nan,
                "exact": exact,
            }
        )
    save_table(
        pd.DataFrame(slopes),
        output / "tables" / "E1_convergence_slopes",
        caption="Median slope of $\\log$ RMSE versus $\\log N$; exact methods have zero error up to the precision of "
        "the reference.",
    )
    plot_error_curves(
        summary,
        "n_samples",
        output / "figures" / "E1_rmse_vs_n",
        "E1: convergence to the reference",
    )


# Order of the non-numeric levels of the regime factors
LEVEL_ORDER = {"S": 0, "both": 1, "R": 2}


def _sort_levels(table: pd.DataFrame) -> pd.DataFrame:
    """Sorts the rows by factor and by the numeric value of the level (or ``LEVEL_ORDER``), so the levels are plotted in
    their natural order (10, 20, 50, 100) instead of the string order (10, 100, 20, 50)"""
    def _key(level) -> float:
        try:
            return float(level)
        except (TypeError, ValueError):
            return float(LEVEL_ORDER.get(str(level), len(LEVEL_ORDER)))

    table = table.copy()
    table["_level_key"] = table["level"].map(_key)
    table = table.sort_values(["factor", "_level_key"], kind="stable").drop(columns="_level_key")
    return table.reset_index(drop=True)


def analyze_e3(results: pd.DataFrame, output: Path) -> None:
    """E3: relative RMSE reduction of each pair at the largest common sample size, and n_eff / N, versus every factor"""
    summary = task_summary(results, ["factor", "level"])
    summary["level"] = summary["level"].astype(str)
    rows = []
    for (factor, level), group in summary.groupby(["factor", "level"]):
        for baseline, proposed in PAIRS:
            wide = group.pivot_table(
                index=["task_id", "n_samples"], columns="method", values="rmse"
            ).reset_index()
            if baseline not in wide or proposed not in wide:
                continue
            n_max = wide.dropna(subset=[baseline, proposed])["n_samples"].max()
            at_n = wide[wide["n_samples"] == n_max]
            budgets = rmse_at_budgets(group, (baseline, proposed))
            last = (
                budgets[budgets["budget_index"] == budgets["budget_index"].max()]
                if len(budgets)
                else budgets
            )
            row = {
                "factor": factor,
                "level": level,
                "pair": f"{proposed} vs {baseline}",
                "n_samples": n_max,
                **{
                    f"equal_N_{k}": v
                    for k, v in paired_comparison(
                        at_n[baseline], at_n[proposed]
                    ).items()
                },
            }
            if len(last):
                row.update(
                    {
                        f"equal_time_{k}": v
                        for k, v in paired_comparison(
                            last[baseline], last[proposed]
                        ).items()
                    }
                )
            if baseline == "LW":
                neff = (
                    group[group["n_samples"] == n_max]
                    .groupby("method")["n_eff"]
                    .median()
                )
                row["neff_frac_LW"] = neff.get("LW", np.nan) / n_max
                row["neff_frac_RBLW"] = neff.get("RBLW", np.nan) / n_max
            rows.append(row)
    table = pd.DataFrame(rows)
    if not len(table):
        return
    for column in ["equal_N_p_value", "equal_time_p_value"]:
        if column in table:
            table[column.replace("p_value", "p_holm")] = holm_correction(
                table[column].to_numpy()
            )
    save_table(
        table,
        output / "tables" / "E3_regimes",
        caption="Relative RMSE reduction per regime factor.",
    )

    table = _sort_levels(table)
    factors = list(dict.fromkeys(table["factor"]))
    # First row: relative RMSE reduction of both pairs; second row: n_eff / N of LW and RBLW
    fig, axes = plt.subplots(
        2, len(factors), figsize=(3.4 * len(factors), 6.2), squeeze=False
    )
    for ax, factor in zip(axes[1], factors):
        by_factor = table[(table["factor"] == factor) & (table["pair"] == "RBLW vs LW")]
        for method in ["LW", "RBLW"]:
            column = f"neff_frac_{method}"
            if column in by_factor:
                ax.plot(
                    by_factor["level"],
                    by_factor[column],
                    marker="o",
                    label=method,
                    color=METHOD_COLORS.get(method),
                )
        ax.set_yscale("log")
        ax.set_xlabel(factor)
        ax.set_ylabel("$n_{\\mathrm{eff}} / N$")
        ax.legend(fontsize=7)
    for ax, factor in zip(axes[0], factors):
        by_factor = table[table["factor"] == factor]
        for pair, group in by_factor.groupby("pair"):
            ax.errorbar(
                group["level"],
                group["equal_N_median_reduction"],
                yerr=[
                    group["equal_N_median_reduction"] - group["equal_N_ci_low"],
                    group["equal_N_ci_high"] - group["equal_N_median_reduction"],
                ],
                marker="o",
                capsize=3,
                label=pair,
            )
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel(factor)
        ax.set_ylabel("Relative RMSE reduction")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output / "figures" / "E3_regimes")


def analyze_e4(results: pd.DataFrame, output: Path) -> None:
    """E4: decomposition of the RBLW gain, block moves and rejection rounds"""
    summary = task_summary(results, ["ablation"])
    summary["ess_per_second"] = summary.get("n_eff", np.nan) / summary["runtime"]
    for ablation, group in summary.groupby("ablation"):
        columns = [
            "rmse",
            "w1",
            "runtime",
            "ess_per_second",
            "max_rhat",
            "mean_exact_rate",
            "mean_acceptance_rate",
            "mean_block_acceptance_rate",
        ]
        columns = [c for c in columns if c in group and group[c].notna().any()]
        table = group.groupby(["method", "n_samples"])[columns].median().reset_index()
        if ablation == "decomposition":
            reference = table[table["method"] == "LW"].set_index("n_samples")["rmse"]
            table["rmse_relative_to_LW"] = table["rmse"] / table["n_samples"].map(
                reference
            )
        save_table(
            table,
            output / "tables" / f"E4_{ablation}",
            caption=f"Ablation: {ablation} (medians over tasks).",
        )
        plot_error_curves(
            group,
            "n_samples",
            output / "figures" / f"E4_{ablation}_rmse_vs_n",
            f"E4: {ablation}",
        )


def analyze_e5(results: pd.DataFrame, output: Path) -> None:
    """E5: setup time a and time per sample b of runtime(N) = a + b N per method and factor level, kernel evaluations per
    sample, and the log-log slope of b versus every numeric factor"""
    rows = []
    for (factor, level, unit_id, method), group in results.groupby(
        ["factor", "level", "unit_id", "method"]
    ):
        means = (
            group.groupby("n_samples")[["runtime", "kernel_evaluations"]]
            .mean()
            .reset_index()
        )
        if len(means) < 2:
            continue
        per_sample, setup = np.polyfit(means["n_samples"], means["runtime"], 1)
        rows.append(
            {
                "factor": factor,
                "level": level,
                "unit_id": unit_id,
                "method": method,
                "setup_time": setup,
                "time_per_sample": per_sample,
                "kernel_evaluations_per_sample": float(
                    (means["kernel_evaluations"] / means["n_samples"]).median()
                ),
            }
        )
    fits = pd.DataFrame(rows)
    if not len(fits):
        return
    table = (
        fits.groupby(["factor", "level", "method"])[
            ["setup_time", "time_per_sample", "kernel_evaluations_per_sample"]
        ]
        .median()
        .reset_index()
    )
    save_table(
        table,
        output / "tables" / "E5_costs",
        caption="Setup time and time per sample (median over networks).",
        float_format="%.3g",
    )
    slopes = []
    for (factor, method), group in table.groupby(["factor", "method"]):
        levels = pd.to_numeric(group["level"], errors="coerce")
        valid = (levels > 0) & (group["time_per_sample"] > 0)
        if valid.sum() >= 2:
            slopes.append(
                {
                    "factor": factor,
                    "method": method,
                    "loglog_slope": np.polyfit(
                        np.log(levels[valid]),
                        np.log(group.loc[valid, "time_per_sample"]),
                        1,
                    )[0],
                }
            )
    if slopes:
        save_table(
            pd.DataFrame(slopes),
            output / "tables" / "E5_slopes",
            caption="Log-log slope of the time per sample versus each factor.",
        )
    factors = list(dict.fromkeys(table["factor"]))
    fig, axes = plt.subplots(
        1, len(factors), figsize=(3.4 * len(factors), 3.2), squeeze=False
    )
    for ax, factor in zip(axes[0], factors):
        for method, group in table[table["factor"] == factor].groupby("method"):
            levels = pd.to_numeric(group["level"], errors="coerce")
            ax.plot(
                levels,
                group["time_per_sample"],
                marker="o",
                label=method,
                color=METHOD_COLORS.get(method),
            )
        ax.set_yscale("log")
        ax.set_xlabel(factor)
        ax.set_ylabel("Time per sample (s)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output / "figures" / "E5_time_per_sample")


def _friedman_and_cd(
    matrix: pd.DataFrame, output: Path, name: str, lower_better: bool
) -> None:
    """Friedman test and critical difference diagram of a (blocks x methods) metric matrix"""
    matrix = matrix.dropna()
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        print(
            f"{name}: not enough complete blocks ({matrix.shape[0]}) for the Friedman test."
        )
        return
    labels = {m: m for m in matrix.columns}
    ranking = get_ranking_matrix(
        matrix,
        labels,
        output / "tables" / f"{name}_ranking.csv",
        lower_better=lower_better,
    )
    statistic, p_value = compute_friedman_test(ranking)
    pd.DataFrame(
        [{"statistic": statistic, "p_value": p_value, "n_blocks": matrix.shape[0]}]
    ).to_csv(output / "tables" / f"{name}_friedman.csv", index=False)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    plot_critical_difference_diagram(
        matrix,
        labels,
        output / "figures" / f"{name}_cd_diagram.png",
        lower_better=lower_better,
        friedman_stat=statistic,
        p_value=p_value,
    )
    plt.close("all")


def analyze_e6(results: pd.DataFrame, output: Path, epsilon: float) -> None:
    """E6: comparison per dataset, log score of the true values, and Friedman test across datasets"""
    analyze_comparison(
        results, output, epsilon, "E6", "E6: SPBNs learned from real data"
    )
    summary = task_summary(results, ["dataset"])
    columns = [c for c in ["rmse", "w1", "log_score", "runtime"] if c in summary]
    table = (
        summary.groupby(["dataset", "method", "n_samples"])[columns]
        .median()
        .reset_index()
    )
    save_table(
        table,
        output / "tables" / "E6_datasets",
        caption="Median errors and log score per dataset.",
    )
    n_max = summary.groupby("method")["n_samples"].max().min()
    matrix = summary[summary["n_samples"] == n_max].pivot_table(
        index="dataset", columns="method", values="rmse"
    )
    _friedman_and_cd(matrix, output, "E6_rmse", lower_better=True)


def analyze_e7(results: pd.DataFrame, output: Path) -> None:
    """E7: classification with missing features per dataset, model and missing rate, and Friedman test on the log-loss"""
    columns = [
        c
        for c in ["log_loss", "brier", "correct", "prob_error", "runtime"]
        if c in results
    ]
    table = (
        results.groupby(["dataset", "model", "missing_rate", "method"])[columns]
        .mean()
        .reset_index()
    )
    table = table.rename(columns={"correct": "accuracy"})
    save_table(
        table,
        output / "tables" / "E7_classification",
        caption="Classification with missing features (means over test instances and repetitions).",
    )
    overall = (
        results.groupby(["missing_rate", "method"])[columns]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    save_table(
        overall,
        output / "tables" / "E7_overall",
        caption="Classification with missing features, all datasets.",
    )
    for metric, lower_better in [("log_loss", True), ("prob_error", True)]:
        if metric not in results:
            continue
        blocks = (
            results.groupby(["dataset", "model", "missing_rate", "method"])[metric]
            .mean()
            .reset_index()
        )
        blocks["block"] = (
            blocks["dataset"]
            + "/"
            + blocks["model"]
            + "/"
            + blocks["missing_rate"].astype(str)
        )
        matrix = blocks.pivot_table(index="block", columns="method", values=metric)
        _friedman_and_cd(matrix, output, f"E7_{metric}", lower_better=lower_better)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric in zip(axes, ["log_loss", "accuracy"]):
        for method, group in overall.groupby("method"):
            ax.plot(
                group["missing_rate"],
                group[metric],
                marker="o",
                label=method,
                color=METHOD_COLORS.get(method),
            )
        ax.set_xlabel("Missing rate")
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output / "figures" / "E7_missing_rate")


# endregion Experiment analyses


def analyze(experiment: str, input_dir: Path, epsilon: float) -> Path:
    output = input_dir / experiment
    results = load_results(output)
    reference_check_table(output, experiment)
    match experiment:
        case "E1":
            analyze_e1(results, output)
        case "E2":
            analyze_comparison(
                results, output, epsilon, "E2", "E2: comparison at a fixed budget"
            )
        case "E3":
            analyze_e3(results, output)
        case "E4":
            analyze_e4(results, output)
        case "E5":
            analyze_e5(results, output)
        case "E6":
            analyze_e6(results, output, epsilon)
        case "E7":
            analyze_e7(results, output)
        case _:
            raise ValueError(f"Unknown experiment {experiment!r}.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "experiment", choices=["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
    )
    parser.add_argument(
        "--quick", action="store_true", help="Analyze the results of a --quick run."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Folder of the experiment results (default data/results/inference).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Target RMSE of the time-to-epsilon comparison.",
    )
    args = parser.parse_args()
    input_dir = (
        Path(args.input_dir)
        if args.input_dir
        else (RESULT_PATH / "quick" if args.quick else RESULT_PATH)
    )
    path = analyze(args.experiment, input_dir, args.epsilon)
    print(f"Figures and tables in {path / 'figures'} and {path / 'tables'}")
