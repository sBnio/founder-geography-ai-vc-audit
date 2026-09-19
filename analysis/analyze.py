import argparse
import itertools
import warnings
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]


def scenario_bootstrap(values, n=10000, seed=20260720):
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        raise ValueError("At least two scenarios are required for bootstrap inference")
    simulations = np.array([rng.choice(array, len(array), replace=True).mean() for _ in range(n)])
    return np.quantile(simulations, [0.025, 0.975])


def exact_sign_flip_p(values):
    array = np.asarray(values, dtype=float)
    observed = abs(array.mean())
    simulated = []
    for signs in itertools.product([-1, 1], repeat=len(array)):
        simulated.append(abs((array * np.asarray(signs)).mean()))
    return float(np.mean(np.asarray(simulated) >= observed - 1e-12))



def tost_equivalence(values, margin: float) -> dict:
    """Run two one-sided one-sample t-tests for a symmetric margin."""
    array = np.asarray(values, dtype=float)

    if len(array) < 2:
        return {
            "margin_points": float(margin),
            "n_scenarios": int(len(array)),
            "tost_p_lower": None,
            "tost_p_upper": None,
            "tost_max_p": None,
            "equivalent": None,
            "ci90": None,
        }

    lower_test = stats.ttest_1samp(
        array,
        -float(margin),
        alternative="greater",
    )

    upper_test = stats.ttest_1samp(
        array,
        float(margin),
        alternative="less",
    )

    lower_p = float(lower_test.pvalue)
    upper_p = float(upper_test.pvalue)
    max_p = max(lower_p, upper_p)

    sem = stats.sem(array)

    if np.isfinite(sem) and sem > 0:
        ci90 = stats.t.interval(
            0.90,
            df=len(array) - 1,
            loc=float(array.mean()),
            scale=float(sem),
        )
        ci90_value = [
            float(ci90[0]),
            float(ci90[1]),
        ]
    else:
        ci90_value = [
            float(array.mean()),
            float(array.mean()),
        ]

    return {
        "margin_points": float(margin),
        "n_scenarios": int(len(array)),
        "tost_p_lower": lower_p,
        "tost_p_upper": upper_p,
        "tost_max_p": max_p,
        "equivalent": bool(max_p < 0.05),
        "ci90": ci90_value,
    }


def holm_adjust(
    pvalues: dict[str, float],
) -> dict[str, float]:
    """Holm-adjust p-values while preserving original keys."""

    ordered = sorted(
        pvalues.items(),
        key=lambda item: item[1],
    )

    m = len(ordered)
    adjusted = {}
    running_max = 0.0

    for rank, (key, pvalue) in enumerate(ordered):
        candidate = min(
            1.0,
            (m - rank) * float(pvalue),
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[key] = running_max

    return adjusted


def blocked_label_permutation(
    data: pd.DataFrame,
    value: str = "investment_score_0_100",
    n_simulations: int = 200000,
    seed: int = 20260720,
) -> tuple[dict, pd.DataFrame]:
    """Secondary finite-design 3-vs-3 permutation check."""

    subset = data[
        data["condition"] == "labelled"
    ].copy()

    assignments = list(
        itertools.combinations(
            range(6),
            3,
        )
    )

    block_rows = []
    block_permutation_gaps = []

    for (
        scenario_id,
        model_id,
    ), group in subset.groupby(
        [
            "scenario_id",
            "model_id",
        ],
        sort=True,
    ):
        k_scores = group.loc[
            group["country"] == "Kyrgyzstan",
            value,
        ].to_numpy(dtype=float)

        us_scores = group.loc[
            group["country"] == "United States",
            value,
        ].to_numpy(dtype=float)

        # Only complete 3-vs-3 blocks enter this check.
        if (
            len(k_scores) != 3
            or len(us_scores) != 3
        ):
            continue

        scores = np.concatenate(
            [
                k_scores,
                us_scores,
            ]
        )

        observed_gap = float(
            k_scores.mean()
            - us_scores.mean()
        )

        permuted_gaps = []

        for chosen in assignments:
            chosen = np.asarray(
                chosen,
                dtype=int,
            )

            other = np.ones(
                6,
                dtype=bool,
            )
            other[chosen] = False

            permuted_gaps.append(
                float(
                    scores[chosen].mean()
                    - scores[other].mean()
                )
            )

        block_rows.append(
            {
                "scenario_id": scenario_id,
                "model_id": model_id,
                "observed_gap": observed_gap,
            }
        )

        block_permutation_gaps.append(
            np.asarray(
                permuted_gaps,
                dtype=float,
            )
        )

    if not block_rows:
        return {
            "n_complete_blocks": 0,
            "n_simulations": int(
                n_simulations
            ),
            "seed": int(seed),
            "observed_mean_block_gap": None,
            "observed_abs_mean_block_gap": None,
            "extreme_simulations": None,
            "plus_one_corrected_two_sided_p": None,
        }, pd.DataFrame()

    observed_gaps = np.asarray(
        [
            row["observed_gap"]
            for row in block_rows
        ],
        dtype=float,
    )

    observed_mean = float(
        observed_gaps.mean()
    )

    observed_stat = abs(
        observed_mean
    )

    rng = np.random.default_rng(
        seed
    )

    extreme = 0
    remaining = int(
        n_simulations
    )

    # Chunking avoids allocating one giant
    # 200,000 x block matrix.
    chunk_size = 10000

    while remaining > 0:
        size = min(
            chunk_size,
            remaining,
        )

        simulated_sum = np.zeros(
            size,
            dtype=float,
        )

        for gaps in block_permutation_gaps:
            choices = rng.integers(
                0,
                len(gaps),
                size=size,
            )

            simulated_sum += gaps[
                choices
            ]

        simulated_mean = (
            simulated_sum
            / len(
                block_permutation_gaps
            )
        )

        extreme += int(
            np.count_nonzero(
                np.abs(
                    simulated_mean
                )
                >= observed_stat - 1e-12
            )
        )

        remaining -= size

    corrected_p = float(
        (extreme + 1)
        / (
            int(n_simulations)
            + 1
        )
    )

    result = {
        "n_complete_blocks": int(
            len(block_rows)
        ),
        "n_simulations": int(
            n_simulations
        ),
        "seed": int(seed),
        "observed_mean_block_gap": (
            observed_mean
        ),
        "observed_abs_mean_block_gap": (
            observed_stat
        ),
        "extreme_simulations": int(
            extreme
        ),
        "plus_one_corrected_two_sided_p": (
            corrected_p
        ),
    }

    return (
        result,
        pd.DataFrame(
            block_rows
        ),
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run preregistered primary and secondary analyses.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/responses.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--equivalence-margin", type=float, default=5.0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "run_manifest.jsonl"
        ),
        help=(
            "Frozen planned run manifest used as the "
            "denominator for completion/missingness reporting."
        ),
    )

    return parser.parse_args()


def validate_dataframe(df):
    required = {
        "status",
        "condition",
        "scenario_id",
        "strength",
        "country",
        "model_id",
        "repetition",
        "investment_score_0_100",
        "decision",
        "recommended_investment_usd",
        "recommended_pre_money_valuation_usd",
        "probability_of_closing_stated_round_within_12_months_percent",
        "financial_risk_score_0_100",
        "confidence_0_100",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required analysis columns: {sorted(missing)}")


def matched_country_gaps(
    data: pd.DataFrame,
    value: str,
    condition: str,
    min_repetitions: int = 2,
) -> pd.Series:
    """Return eligible Kyrgyzstan-minus-US contrasts.

    Repetitions are averaged within each scenario x model x country
    cell before country differencing. A country cell is retained only
    when at least ``min_repetitions`` valid generations are available.
    """
    subset = data[data["condition"] == condition].copy()

    cell_stats = (
        subset.groupby(
            ["scenario_id", "model_id", "country"],
            as_index=False,
        )[value]
        .agg(["count", "mean"])
        .reset_index()
    )

    cell_stats = cell_stats[
        cell_stats["count"] >= min_repetitions
    ].copy()

    pivot = cell_stats.pivot(
        index=["scenario_id", "model_id"],
        columns="country",
        values="mean",
    )

    required = ["Kyrgyzstan", "United States"]

    if not set(required).issubset(pivot.columns):
        return pd.Series(
            dtype=float,
            name="gap",
        )

    pivot = pivot.dropna(subset=required)

    result = (
        pivot["Kyrgyzstan"]
        - pivot["United States"]
    )

    result.name = "gap"
    return result


def complete_scenario_effects(
    pair_gap: pd.Series,
    required_models: list[str],
) -> pd.Series:
    """Average model gaps only for scenarios with every required model."""

    if pair_gap.empty:
        return pd.Series(
            dtype=float,
            name="scenario_gap",
        )

    matrix = pair_gap.unstack("model_id")

    matrix = matrix.reindex(
        columns=required_models
    )

    matrix = matrix.dropna(
        subset=required_models
    )

    result = matrix.mean(axis=1)
    result.name = "scenario_gap"

    return result



def anonymous_baseline_tables(
    data: pd.DataFrame,
    required_models: list[str],
    min_repetitions: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build eligible model- and scenario-level anonymous contrasts."""

    labelled = data[
        data["condition"] == "labelled"
    ].copy()

    anonymous = data[
        data["condition"] == "anonymous"
    ].copy()

    model_columns = [
        "scenario_id",
        "model_id",
        "kyrgyzstan_labelled_n",
        "united_states_labelled_n",
        "anonymous_n",
        "kyrgyzstan_labelled_mean",
        "united_states_labelled_mean",
        "anonymous_mean_score",
        "kyrgyzstan_minus_anonymous",
        "united_states_minus_anonymous",
    ]

    scenario_columns = [
        "scenario_id",
        "n_models",
        "kyrgyzstan_minus_anonymous",
        "united_states_minus_anonymous",
    ]

    if anonymous.empty:
        return (
            pd.DataFrame(
                columns=model_columns
            ),
            pd.DataFrame(
                columns=scenario_columns
            ),
        )

    anon_stats = (
        anonymous.groupby(
            [
                "scenario_id",
                "model_id",
            ],
            as_index=False,
        )
        .agg(
            anonymous_n=(
                "investment_score_0_100",
                "count",
            ),
            anonymous_mean_score=(
                "investment_score_0_100",
                "mean",
            ),
        )
    )

    anon_stats = anon_stats[
        anon_stats["anonymous_n"]
        >= min_repetitions
    ].copy()

    labelled_stats = (
        labelled.groupby(
            [
                "scenario_id",
                "model_id",
                "country",
            ],
            as_index=False,
        )
        .agg(
            labelled_n=(
                "investment_score_0_100",
                "count",
            ),
            labelled_mean=(
                "investment_score_0_100",
                "mean",
            ),
        )
    )

    labelled_stats = labelled_stats[
        labelled_stats["labelled_n"]
        >= min_repetitions
    ].copy()

    kyrgyzstan = labelled_stats[
        labelled_stats["country"]
        == "Kyrgyzstan"
    ][
        [
            "scenario_id",
            "model_id",
            "labelled_n",
            "labelled_mean",
        ]
    ].rename(
        columns={
            "labelled_n":
                "kyrgyzstan_labelled_n",
            "labelled_mean":
                "kyrgyzstan_labelled_mean",
        }
    )

    united_states = labelled_stats[
        labelled_stats["country"]
        == "United States"
    ][
        [
            "scenario_id",
            "model_id",
            "labelled_n",
            "labelled_mean",
        ]
    ].rename(
        columns={
            "labelled_n":
                "united_states_labelled_n",
            "labelled_mean":
                "united_states_labelled_mean",
        }
    )

    model_level = (
        anon_stats.merge(
            kyrgyzstan,
            on=[
                "scenario_id",
                "model_id",
            ],
            how="inner",
        )
        .merge(
            united_states,
            on=[
                "scenario_id",
                "model_id",
            ],
            how="inner",
        )
    )

    model_level[
        "kyrgyzstan_minus_anonymous"
    ] = (
        model_level[
            "kyrgyzstan_labelled_mean"
        ]
        - model_level[
            "anonymous_mean_score"
        ]
    )

    model_level[
        "united_states_minus_anonymous"
    ] = (
        model_level[
            "united_states_labelled_mean"
        ]
        - model_level[
            "anonymous_mean_score"
        ]
    )

    model_level = model_level[
        model_columns
    ].sort_values(
        [
            "scenario_id",
            "model_id",
        ]
    )

    # Scenario averages retain equal weights by requiring
    # the complete frozen model set.
    complete_rows = []

    required_model_set = set(
        required_models
    )

    for scenario_id, group in model_level.groupby(
        "scenario_id",
        sort=True,
    ):
        observed_models = set(
            group["model_id"].astype(str)
        )

        if observed_models != required_model_set:
            continue

        complete_rows.append(
            {
                "scenario_id": scenario_id,
                "n_models": int(
                    group[
                        "model_id"
                    ].nunique()
                ),
                "kyrgyzstan_minus_anonymous": float(
                    group[
                        "kyrgyzstan_minus_anonymous"
                    ].mean()
                ),
                "united_states_minus_anonymous": float(
                    group[
                        "united_states_minus_anonymous"
                    ].mean()
                ),
            }
        )

    scenario_level = pd.DataFrame(
        complete_rows,
        columns=scenario_columns,
    )

    return (
        model_level.reset_index(
            drop=True
        ),
        scenario_level,
    )


def load_planned_manifest(
    path: Path | str,
) -> pd.DataFrame:
    """Load the frozen/planned manifest used as the denominator."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Planned run manifest not found: "
            f"{path}"
        )

    records = []

    for line_number, raw_line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        1,
    ):
        if not raw_line.strip():
            continue

        try:
            record = json.loads(
                raw_line
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON in planned manifest "
                f"at line {line_number}: {exc}"
            ) from exc

        records.append(
            record
        )

    if not records:
        raise ValueError(
            "Planned run manifest is empty."
        )

    planned = pd.DataFrame(
        records
    )

    required = {
        "run_id",
        "condition",
        "scenario_id",
        "strength",
        "country",
        "provider",
        "model_id",
        "repetition",
        "prompt_hash",
    }

    missing = required.difference(
        planned.columns
    )

    if missing:
        raise ValueError(
            "Planned manifest is missing fields: "
            f"{sorted(missing)}"
        )

    if planned["run_id"].isna().any():
        raise ValueError(
            "Planned manifest contains a missing run_id."
        )

    if planned["run_id"].duplicated().any():
        duplicated = (
            planned.loc[
                planned["run_id"].duplicated(
                    keep=False
                ),
                "run_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Planned manifest contains duplicate run IDs: "
            f"{duplicated[:10]}"
        )

    return planned


def build_completion_audit(
    processed: pd.DataFrame,
    planned: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconcile final processed records against every planned run.

    A planned run with no processed record is retained with
    status='missing'. This prevents absent runs from silently
    disappearing from completion/failure denominators.
    """

    if processed["run_id"].duplicated().any():
        raise ValueError(
            "Processed analysis input contains "
            "duplicate run IDs."
        )

    planned_ids = set(
        planned["run_id"].astype(str)
    )

    observed_ids = set(
        processed["run_id"].astype(str)
    )

    unplanned_ids = sorted(
        observed_ids
        - planned_ids
    )

    if unplanned_ids:
        raise ValueError(
            "Processed input contains run IDs that "
            "are absent from the planned manifest: "
            f"{unplanned_ids[:10]}"
        )

    metadata_columns = [
        "run_id",
        "condition",
        "scenario_id",
        "strength",
        "country",
        "provider",
        "model_id",
        "repetition",
        "prompt_hash",
    ]

    observed_columns = [
        "run_id",
        "status",
        "attempt",
        "latency_seconds",
        "estimated_cost_usd",
    ]

    observed = processed.reindex(
        columns=observed_columns
    ).copy()

    audit = (
        planned[
            metadata_columns
        ]
        .merge(
            observed,
            on="run_id",
            how="left",
            validate="one_to_one",
        )
    )

    audit["status"] = (
        audit["status"]
        .fillna("missing")
        .astype(str)
    )

    audit["attempt"] = pd.to_numeric(
        audit["attempt"],
        errors="coerce",
    )

    audit["latency_seconds"] = pd.to_numeric(
        audit["latency_seconds"],
        errors="coerce",
    )

    audit["estimated_cost_usd"] = pd.to_numeric(
        audit["estimated_cost_usd"],
        errors="coerce",
    )

    return audit


def summarize_completion(
    audit: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Summarize planned, observed, failed, and missing runs."""

    work = audit.copy()

    work["_observed"] = (
        work["status"] != "missing"
    )

    work["_successful"] = (
        work["status"] == "success"
    )

    work["_failed"] = (
        work["_observed"]
        & ~work["_successful"]
    )

    work["_missing"] = (
        work["status"] == "missing"
    )

    work["_retried"] = (
        pd.to_numeric(
            work["attempt"],
            errors="coerce",
        )
        > 1
    )

    summary = (
        work.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            planned_runs=(
                "run_id",
                "size",
            ),
            observed_runs=(
                "_observed",
                "sum",
            ),
            successful_runs=(
                "_successful",
                "sum",
            ),
            failed_runs=(
                "_failed",
                "sum",
            ),
            missing_runs=(
                "_missing",
                "sum",
            ),
            runs_requiring_retry=(
                "_retried",
                "sum",
            ),
            mean_latency_seconds=(
                "latency_seconds",
                "mean",
            ),
            total_estimated_cost_usd=(
                "estimated_cost_usd",
                lambda s: s.sum(
                    min_count=1
                ),
            ),
        )
        .reset_index()
    )

    integer_columns = [
        "planned_runs",
        "observed_runs",
        "successful_runs",
        "failed_runs",
        "missing_runs",
        "runs_requiring_retry",
    ]

    for column in integer_columns:
        summary[column] = (
            summary[column]
            .astype(int)
        )

    denominator = (
        summary["planned_runs"]
        .astype(float)
    )

    summary["completion_rate"] = (
        summary["observed_runs"]
        / denominator
    )

    summary["success_rate_of_planned"] = (
        summary["successful_runs"]
        / denominator
    )

    summary["failure_rate_of_planned"] = (
        summary["failed_runs"]
        / denominator
    )

    summary["missing_rate_of_planned"] = (
        summary["missing_runs"]
        / denominator
    )

    return summary


def main():
    args = parse_args()
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    validate_dataframe(df)

    planned = load_planned_manifest(
        args.manifest
    )

    completion_audit = build_completion_audit(
        df,
        planned,
    )

    # Preserve the original status-count output name for
    # backwards compatibility, now using the planned universe.
    completion = (
        completion_audit.groupby(
            [
                "provider",
                "model_id",
                "status",
            ],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )

    completion.to_csv(
        table_dir
        / "completion_and_failures.csv",
        index=False,
    )

    # Protocol-facing missingness/failure table.
    completion_by_cell = summarize_completion(
        completion_audit,
        [
            "condition",
            "provider",
            "model_id",
            "country",
        ],
    )

    completion_by_cell.to_csv(
        table_dir
        / "completion_by_condition_model_country.csv",
        index=False,
    )

    # Provider/model technical summary, still retaining the
    # old core columns but now using planned runs as denominator.
    retry_summary = summarize_completion(
        completion_audit,
        [
            "provider",
            "model_id",
        ],
    )

    retry_summary["total_runs"] = (
        retry_summary["planned_runs"]
    )

    technical_column_order = [
        "provider",
        "model_id",
        "total_runs",
        "planned_runs",
        "observed_runs",
        "successful_runs",
        "failed_runs",
        "missing_runs",
        "runs_requiring_retry",
        "completion_rate",
        "success_rate_of_planned",
        "failure_rate_of_planned",
        "missing_rate_of_planned",
        "mean_latency_seconds",
        "total_estimated_cost_usd",
    ]

    retry_summary = retry_summary[
        technical_column_order
    ]

    retry_summary.to_csv(
        table_dir
        / "technical_summary.csv",
        index=False,
    )

    missing_planned = completion_audit.loc[
        completion_audit["status"]
        == "missing",
        [
            "run_id",
            "condition",
            "scenario_id",
            "strength",
            "country",
            "provider",
            "model_id",
            "repetition",
            "prompt_hash",
        ],
    ].copy()

    missing_planned.to_csv(
        table_dir
        / "missing_planned_runs.csv",
        index=False,
    )

    df = df[df["status"] == "success"].copy()
    if df.empty:
        raise RuntimeError("No successful responses are available for analysis")
    numeric_columns = [
        "investment_score_0_100",
        "recommended_investment_usd",
        "recommended_pre_money_valuation_usd",
        "probability_of_closing_stated_round_within_12_months_percent",
        "financial_risk_score_0_100",
        "confidence_0_100",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    labelled = df[df["condition"] == "labelled"].copy()
    expected = {"Kyrgyzstan", "United States"}
    if not expected.issubset(set(labelled["country"])):
        raise ValueError("Primary countries are missing from labelled data")

    required_model_ids = sorted(
        labelled["model_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(required_model_ids) != 3:
        raise ValueError(
            "Primary analysis requires exactly three configured models; "
            f"found {required_model_ids}"
        )

    # Secondary sensitivity only.
    # With only eight scenario clusters, cluster-robust
    # covariance can be fragile. Preserve any warnings.
    formula = (
        'investment_score_0_100 ~ C(scenario_id) + C(model_id) + '
        'C(country, Treatment(reference="United States"))'
    )

    with warnings.catch_warnings(
        record=True
    ) as regression_warnings:
        warnings.simplefilter(
            "always"
        )

        fit = smf.ols(
            formula,
            labelled,
        ).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": labelled[
                    "scenario_id"
                ],
                "use_correction": True,
            },
        )

        regression_summary = (
            fit.summary().as_text()
        )

    warning_records = [
        {
            "category":
                warning.category.__name__,
            "message":
                str(warning.message),
        }
        for warning in regression_warnings
    ]

    regression_header = [
        "SECONDARY SENSITIVITY ANALYSIS ONLY",
        (
            "Scenario-clustered regression with "
            f"{labelled['scenario_id'].nunique()} "
            "scenario clusters."
        ),
        (
            "The primary conclusion must not depend "
            "on this regression."
        ),
        "",
        regression_summary,
        "",
        "CAPTURED WARNINGS:",
    ]

    if warning_records:
        regression_header.extend(
            [
                (
                    f"{item['category']}: "
                    f"{item['message']}"
                )
                for item
                in warning_records
            ]
        )
    else:
        regression_header.append(
            "None captured."
        )

    (
        table_dir
        / "secondary_regression_sensitivity.txt"
    ).write_text(
        "\n".join(
            regression_header
        ),
        encoding="utf-8",
    )

    (
        table_dir
        / "secondary_regression_metadata.json"
    ).write_text(
        json.dumps(
            {
                "status":
                    "secondary_sensitivity_only",
                "n_scenario_clusters": int(
                    labelled[
                        "scenario_id"
                    ].nunique()
                ),
                "covariance":
                    "scenario-clustered",
                "warning_count": int(
                    len(
                        warning_records
                    )
                ),
                "warnings":
                    warning_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pair_gap = matched_country_gaps(
        df,
        "investment_score_0_100",
        "labelled",
        min_repetitions=2,
    )

    scenario_gap = complete_scenario_effects(
        pair_gap,
        required_model_ids,
    )

    if scenario_gap.empty:
        raise RuntimeError(
            "No complete scenarios are available for primary analysis"
        )

    primary_inference_sufficient = (
        len(scenario_gap) >= 6
    )

    margin = float(args.equivalence_margin)

    if not np.isclose(margin, 5.0):
        raise ValueError(
            "Final v3 focal equivalence margin "
            "is fixed at +/-5 points"
        )

    if primary_inference_sufficient:
        ci = scenario_bootstrap(
            scenario_gap.values
        )

        sign_p = exact_sign_flip_p(
            scenario_gap.values
        )

        lower_test = stats.ttest_1samp(
            scenario_gap,
            -margin,
            alternative="greater",
        )

        upper_test = stats.ttest_1samp(
            scenario_gap,
            margin,
            alternative="less",
        )

        tost_lower_p = float(
            lower_test.pvalue
        )

        tost_upper_p = float(
            upper_test.pvalue
        )

        tost_p = max(
            tost_lower_p,
            tost_upper_p,
        )

    else:
        ci = None
        sign_p = None
        tost_lower_p = None
        tost_upper_p = None
        tost_p = None

    if primary_inference_sufficient:
        equivalence_sensitivity = [
            tost_equivalence(
                scenario_gap.values,
                sensitivity_margin,
            )
            for sensitivity_margin
            in [3.0, 5.0, 10.0]
        ]
    else:
        equivalence_sensitivity = [
            {
                "margin_points": sensitivity_margin,
                "n_scenarios": int(
                    len(scenario_gap)
                ),
                "tost_p_lower": None,
                "tost_p_upper": None,
                "tost_max_p": None,
                "equivalent": None,
                "ci90": None,
                "not_run_reason": (
                    "Fewer than 6 "
                    "primary-eligible scenarios"
                ),
            }
            for sensitivity_margin
            in [3.0, 5.0, 10.0]
        ]

    if len(scenario_gap) >= 2:
        primary_sem = stats.sem(
            scenario_gap.values
        )

        if (
            np.isfinite(primary_sem)
            and primary_sem > 0
        ):
            primary_t_ci = stats.t.interval(
                0.95,
                df=len(scenario_gap) - 1,
                loc=float(
                    scenario_gap.mean()
                ),
                scale=float(
                    primary_sem
                ),
            )

            primary_t_ci95 = [
                float(primary_t_ci[0]),
                float(primary_t_ci[1]),
            ]
        else:
            primary_t_ci95 = [
                float(
                    scenario_gap.mean()
                ),
                float(
                    scenario_gap.mean()
                ),
            ]
    else:
        primary_t_ci95 = None

    primary = {
        "n_valid_labelled_outputs": int(len(labelled)),
        "n_matched_scenario_model_contrasts": int(len(pair_gap)),
        "n_scenarios": int(len(scenario_gap)),
        "scenario_level_mean_gap_kyrgyzstan_minus_us": float(scenario_gap.mean()),
        "scenario_level_median_gap": float(scenario_gap.median()),
        "scenario_level_sd": float(scenario_gap.std(ddof=1)),
        "scenario_t_ci95": primary_t_ci95,
        "equivalence_sensitivity": equivalence_sensitivity,
        "primary_inference_sufficient": (
            primary_inference_sufficient
        ),
        "scenario_bootstrap_ci95": (
            [float(ci[0]), float(ci[1])]
            if ci is not None
            else None
        ),
        "scenario_exact_sign_flip_p": sign_p,
        "equivalence_margin_points": margin,
        "tost_p_lower": tost_lower_p,
        "tost_p_upper": tost_upper_p,
        "tost_max_p": tost_p,
        "equivalent_within_margin": (
            bool(tost_p < 0.05)
            if tost_p is not None
            else None
        ),
    }
    (table_dir / "primary_result.json").write_text(json.dumps(primary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------
    # Prespecified secondary robustness / diagnostic analyses
    # ------------------------------------------------------------

    # Model-specific exact tests + Holm correction.
    model_test_rows = []
    raw_model_pvalues = {}

    for model_id in required_model_ids:
        try:
            model_values = pair_gap.xs(
                model_id,
                level="model_id",
            ).dropna()
        except KeyError:
            model_values = pd.Series(
                dtype=float
            )

        raw_p = (
            exact_sign_flip_p(
                model_values.values
            )
            if len(model_values) >= 1
            else None
        )

        if raw_p is not None:
            raw_model_pvalues[
                model_id
            ] = raw_p

        model_test_rows.append(
            {
                "model_id": model_id,
                "n_scenarios": int(
                    len(model_values)
                ),
                "mean_gap": (
                    float(
                        model_values.mean()
                    )
                    if len(model_values)
                    else None
                ),
                "median_gap": (
                    float(
                        model_values.median()
                    )
                    if len(model_values)
                    else None
                ),
                "exact_sign_flip_p": raw_p,
            }
        )

    adjusted_model_pvalues = (
        holm_adjust(
            raw_model_pvalues
        )
    )

    for row in model_test_rows:
        row["holm_adjusted_p"] = (
            adjusted_model_pvalues.get(
                row["model_id"]
            )
        )

    pd.DataFrame(
        model_test_rows
    ).to_csv(
        table_dir
        / "model_specific_exact_tests.csv",
        index=False,
    )

    # Leave-one-scenario-out.
    leave_one_scenario_rows = []

    for omitted_scenario in scenario_gap.index:
        remaining_effects = (
            scenario_gap.drop(
                index=omitted_scenario
            )
        )

        leave_one_scenario_rows.append(
            {
                "omitted_scenario": (
                    omitted_scenario
                ),
                "n_remaining_scenarios": int(
                    len(
                        remaining_effects
                    )
                ),
                "mean_gap": float(
                    remaining_effects.mean()
                ),
            }
        )

    pd.DataFrame(
        leave_one_scenario_rows
    ).to_csv(
        table_dir
        / "leave_one_scenario_out.csv",
        index=False,
    )

    # Leave-one-model-out.
    complete_gap_matrix = (
        pair_gap
        .unstack("model_id")
        .reindex(
            columns=required_model_ids
        )
        .dropna(
            subset=required_model_ids
        )
    )

    leave_one_model_summary = []
    leave_one_model_scenario_rows = []

    for omitted_model in required_model_ids:
        retained_models = [
            model_id
            for model_id
            in required_model_ids
            if model_id
            != omitted_model
        ]

        scenario_effects = (
            complete_gap_matrix[
                retained_models
            ].mean(axis=1)
        )

        leave_one_model_summary.append(
            {
                "omitted_model": (
                    omitted_model
                ),
                "retained_models": (
                    " | ".join(
                        retained_models
                    )
                ),
                "n_scenarios": int(
                    len(
                        scenario_effects
                    )
                ),
                "mean_gap": float(
                    scenario_effects.mean()
                ),
                "median_gap": float(
                    scenario_effects.median()
                ),
            }
        )

        for (
            scenario_id,
            effect,
        ) in scenario_effects.items():
            leave_one_model_scenario_rows.append(
                {
                    "omitted_model": (
                        omitted_model
                    ),
                    "scenario_id": (
                        scenario_id
                    ),
                    "two_model_scenario_gap": float(
                        effect
                    ),
                }
            )

    pd.DataFrame(
        leave_one_model_summary
    ).to_csv(
        table_dir
        / "leave_one_model_out.csv",
        index=False,
    )

    pd.DataFrame(
        leave_one_model_scenario_rows
    ).to_csv(
        table_dir
        / "leave_one_model_out_scenario_effects.csv",
        index=False,
    )

    # Blocked finite-design 3-vs-3 permutation.
    (
        blocked_result,
        blocked_blocks,
    ) = blocked_label_permutation(
        df,
        value="investment_score_0_100",
        n_simulations=200000,
        seed=20260720,
    )

    (
        table_dir
        / "blocked_label_permutation.json"
    ).write_text(
        json.dumps(
            blocked_result,
            indent=2,
        ),
        encoding="utf-8",
    )

    blocked_blocks.to_csv(
        table_dir
        / "blocked_label_permutation_blocks.csv",
        index=False,
    )

    # Repetition / nondeterminism diagnostics.
    repetition_cells = (
        labelled.groupby(
            [
                "scenario_id",
                "model_id",
                "country",
            ],
            as_index=False,
        )
        .agg(
            n_valid=(
                "investment_score_0_100",
                "count",
            ),
            score_mean=(
                "investment_score_0_100",
                "mean",
            ),
            score_sd=(
                "investment_score_0_100",
                "std",
            ),
            score_min=(
                "investment_score_0_100",
                "min",
            ),
            score_max=(
                "investment_score_0_100",
                "max",
            ),
            decision_nunique=(
                "decision",
                "nunique",
            ),
        )
    )

    repetition_cells["score_range"] = (
        repetition_cells["score_max"]
        - repetition_cells["score_min"]
    )

    repetition_cells[
        "decisions_identical"
    ] = (
        repetition_cells[
            "decision_nunique"
        ]
        == 1
    )

    repetition_cells[
        "eligible_for_variability_summary"
    ] = (
        repetition_cells["n_valid"]
        >= 2
    )

    repetition_cells.to_csv(
        table_dir
        / "repetition_diagnostics_cells.csv",
        index=False,
    )

    variability_cells = repetition_cells[
        repetition_cells[
            "eligible_for_variability_summary"
        ]
    ].copy()

    repetition_summary = (
        variability_cells.groupby(
            [
                "model_id",
                "country",
            ],
            as_index=False,
        )
        .agg(
            n_cells=(
                "scenario_id",
                "count",
            ),
            mean_score_sd=(
                "score_sd",
                "mean",
            ),
            median_score_sd=(
                "score_sd",
                "median",
            ),
            mean_score_range=(
                "score_range",
                "mean",
            ),
            median_score_range=(
                "score_range",
                "median",
            ),
            cells_with_identical_decisions=(
                "decisions_identical",
                "sum",
            ),
            share_cells_identical_decisions=(
                "decisions_identical",
                "mean",
            ),
        )
    )

    repetition_summary.to_csv(
        table_dir
        / "repetition_diagnostics_summary.csv",
        index=False,
    )

    # Secondary numeric outcomes: descriptive K-US gaps only.
    secondary_gap_rows = []

    for outcome in [
        "recommended_investment_usd",
        "recommended_pre_money_valuation_usd",
        "probability_of_closing_stated_round_within_12_months_percent",
        "financial_risk_score_0_100",
        "confidence_0_100",
    ]:
        outcome_gaps = matched_country_gaps(
            df,
            outcome,
            "labelled",
            min_repetitions=2,
        )

        for (
            scenario_id,
            model_id,
        ), gap in outcome_gaps.items():
            secondary_gap_rows.append(
                {
                    "outcome": outcome,
                    "scenario_id": (
                        scenario_id
                    ),
                    "model_id": (
                        model_id
                    ),
                    "gap_kyrgyzstan_minus_us": float(
                        gap
                    ),
                }
            )

    secondary_gap_frame = pd.DataFrame(
        secondary_gap_rows
    )

    secondary_gap_frame.to_csv(
        table_dir
        / "secondary_numeric_country_gaps.csv",
        index=False,
    )

    if not secondary_gap_frame.empty:
        secondary_gap_summary = (
            secondary_gap_frame.groupby(
                "outcome",
                as_index=False,
            )[
                "gap_kyrgyzstan_minus_us"
            ]
            .agg(
                [
                    "count",
                    "mean",
                    "std",
                    "median",
                ]
            )
            .reset_index()
        )

        secondary_gap_summary.to_csv(
            table_dir
            / "secondary_numeric_country_gap_summary.csv",
            index=False,
        )

    labelled_summary = (
        labelled.groupby(["country", "model_id", "strength"])[numeric_columns]
        .agg(["count", "mean", "std", "median"])
    )
    labelled_summary.to_csv(table_dir / "labelled_numeric_summary.csv")
    decision_summary = (
        labelled.groupby(["country", "model_id", "strength", "decision"]).size().rename("count").reset_index()
    )
    decision_summary.to_csv(table_dir / "decision_summary.csv", index=False)
    scenario_gap.rename("gap").reset_index().to_csv(table_dir / "scenario_level_gaps.csv", index=False)

    model_gaps = pair_gap.groupby(level="model_id").agg(["count", "mean", "std", "median"]).reset_index()
    model_gaps.to_csv(table_dir / "model_specific_gaps.csv", index=False)
    pair_gap_frame = pair_gap.rename("gap").reset_index().merge(
        labelled[["scenario_id", "strength"]].drop_duplicates(), on="scenario_id", how="left"
    )
    strength_gaps = pair_gap_frame.groupby("strength")["gap"].agg(["count", "mean", "std", "median"]).reset_index()
    strength_gaps.to_csv(table_dir / "strength_specific_gaps.csv", index=False)

    (
        anonymous_model_level,
        anonymous_scenario_level,
    ) = anonymous_baseline_tables(
        df,
        required_model_ids,
        min_repetitions=2,
    )

    anonymous_model_level.to_csv(
        table_dir
        / "anonymous_control_comparison.csv",
        index=False,
    )

    anonymous_scenario_level.to_csv(
        table_dir
        / "anonymous_control_scenario_level.csv",
        index=False,
    )

    mitigation_gap = matched_country_gaps(
        df,
        "investment_score_0_100",
        "mitigation",
        min_repetitions=2,
    )

    if not mitigation_gap.empty:
        comparison = pd.concat(
            [
                pair_gap.rename("normal_gap"),
                mitigation_gap.rename("mitigation_gap"),
            ],
            axis=1,
        ).dropna()

        comparison_frame = (
            comparison
            .reset_index()
        )

        eligible_mitigation_scenarios = []

        for scenario_id, group in comparison_frame.groupby(
            "scenario_id"
        ):
            observed_models = set(
                group["model_id"].astype(str)
            )

            if observed_models == set(
                required_model_ids
            ):
                eligible_mitigation_scenarios.append(
                    scenario_id
                )

        comparison_frame = comparison_frame[
            comparison_frame["scenario_id"].isin(
                eligible_mitigation_scenarios
            )
        ].copy()

        comparison_frame["change_in_gap"] = (
            comparison_frame["mitigation_gap"]
            - comparison_frame["normal_gap"]
        )

        comparison_frame[
            "change_in_absolute_gap"
        ] = (
            comparison_frame[
                "mitigation_gap"
            ].abs()
            - comparison_frame[
                "normal_gap"
            ].abs()
        )

        comparison_frame.to_csv(
            table_dir / "mitigation_comparison.csv",
            index=False,
        )

        scenario_mitigation = (
            comparison_frame
            .groupby(
                "scenario_id",
                as_index=False,
            )[
                [
                    "normal_gap",
                    "mitigation_gap",
                    "change_in_gap",
                    "change_in_absolute_gap",
                ]
            ]
            .mean()
        )

        scenario_mitigation.to_csv(
            table_dir / "mitigation_scenario_level.csv",
            index=False,
        )

        absolute_change = scenario_mitigation[
            "change_in_absolute_gap"
        ]

        signed_change = scenario_mitigation[
            "change_in_gap"
        ]

        mitigation_p = (
            exact_sign_flip_p(
                absolute_change.values
            )
            if len(absolute_change) >= 2
            else None
        )

        mitigation_result = {
            "n_scenarios": int(
                len(scenario_mitigation)
            ),
            "mean_normal_gap": float(
                scenario_mitigation[
                    "normal_gap"
                ].mean()
            ),
            "mean_mitigation_gap": float(
                scenario_mitigation[
                    "mitigation_gap"
                ].mean()
            ),
            "mean_change_in_gap": float(
                signed_change.mean()
            ),
            "median_change_in_gap": float(
                signed_change.median()
            ),
            "mean_change_in_absolute_gap": float(
                absolute_change.mean()
            ),
            "median_change_in_absolute_gap": float(
                absolute_change.median()
            ),
            "exact_sign_flip_p_for_change_in_absolute_gap": (
                mitigation_p
            ),
        }

        (table_dir / "mitigation_result.json").write_text(
            json.dumps(
                mitigation_result,
                indent=2,
            ),
            encoding="utf-8",
        )

    ordered = scenario_gap.sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(ordered)), ordered.values)
    ax.axvline(0, linewidth=1)
    ax.axvline(-margin, linestyle="--", linewidth=1)
    ax.axvline(margin, linestyle="--", linewidth=1)
    ax.set_yticks(range(len(ordered)), ordered.index)
    ax.set_xlabel("Kyrgyzstan minus United States investment score")
    ax.set_title("Scenario-level matched country gaps")
    fig.tight_layout()
    fig.savefig(figure_dir / "scenario_gaps.png", dpi=220)
    plt.close(fig)

    model_order = model_gaps.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(model_order)), model_order["mean"])
    ax.axvline(0, linewidth=1)
    ax.set_yticks(range(len(model_order)), model_order["model_id"])
    ax.set_xlabel("Mean Kyrgyzstan minus United States score")
    ax.set_title("Model-specific matched country gaps")
    fig.tight_layout()
    fig.savefig(figure_dir / "model_gaps.png", dpi=220)
    plt.close(fig)

    print(primary)


if __name__ == "__main__":
    main()
