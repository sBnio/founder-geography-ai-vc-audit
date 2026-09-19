from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DIMENSIONS = {
    "geography_present": {"yes", "no"},
    "evidence_status": {
        "none",
        "neutral_restatement",
        "supported",
        "unsupported_general_inference",
        "unsupplied_specific_geographic_claim",
    },
    "direction": {
        "none",
        "neutral",
        "negative",
        "positive",
        "mixed",
    },
    "decision_relevance": {
        "no",
        "yes",
        "unclear",
    },
    "coder_confidence": {
        "high",
        "medium",
        "low",
    },
}

DERIVED_VARIABLES = [
    "unsupported_geography",
    "unsupported_negative",
    "unsupported_positive",
    "geography_decision_relevant",
]

VALID_CONDITIONS = {
    "labelled",
    "anonymous",
    "mitigation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create final qualitative derived variables "
            "and prespecified descriptive summaries "
            "from completed human coding."
        )
    )

    parser.add_argument(
        "--researcher",
        type=Path,
        default=(
            ROOT
            / "data/coding/researcher_blinded_full.csv"
        ),
        help="Completed researcher coding file.",
    )

    parser.add_argument(
        "--private-key",
        type=Path,
        default=(
            ROOT
            / "data/coding/PRIVATE_coding_key.csv"
        ),
        help=(
            "Private metadata key produced when "
            "blinded coding sheets were created."
        ),
    )

    parser.add_argument(
        "--adjudication",
        type=Path,
        default=(
            ROOT
            / "data/coding/adjudication_sheet.csv"
        ),
        help=(
            "Completed adjudication sheet. "
            "Used only when it exists."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/qualitative"
        ),
    )

    return parser.parse_args()


def normalize(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def load_researcher(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Researcher coding file not found: {path}"
        )

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    required = [
        "coding_id",
        *DIMENSIONS.keys(),
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Researcher file missing required "
            f"columns: {missing}"
        )

    frame = frame[required].copy()

    frame["coding_id"] = (
        frame["coding_id"]
        .astype(str)
        .str.strip()
    )

    if frame["coding_id"].eq("").any():
        raise ValueError(
            "Researcher file contains blank coding_id."
        )

    if frame["coding_id"].duplicated().any():
        duplicates = (
            frame.loc[
                frame["coding_id"].duplicated(
                    keep=False
                ),
                "coding_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Researcher file contains duplicate "
            f"coding_id values: {duplicates[:10]}"
        )

    for dimension, allowed in DIMENSIONS.items():
        frame[dimension] = normalize(
            frame[dimension]
        )

        blank = frame[
            frame[dimension].eq("")
        ]

        if not blank.empty:
            raise ValueError(
                f"Blank researcher values for "
                f"{dimension}: "
                f"{blank['coding_id'].tolist()[:10]}"
            )

        invalid = frame[
            ~frame[dimension].isin(
                allowed
            )
        ]

        if not invalid.empty:
            examples = (
                invalid[
                    [
                        "coding_id",
                        dimension,
                    ]
                ]
                .head(10)
                .to_dict(
                    orient="records"
                )
            )

            raise ValueError(
                f"Invalid values for {dimension}. "
                f"Allowed: {sorted(allowed)}. "
                f"Examples: {examples}"
            )

    return frame


def validate_consistency(
    frame: pd.DataFrame,
) -> None:
    bad_no_geo = frame[
        frame["geography_present"].eq("no")
        & (
            ~frame["evidence_status"].eq("none")
            | ~frame["direction"].eq("none")
            | ~frame[
                "decision_relevance"
            ].eq("no")
        )
    ]

    if not bad_no_geo.empty:
        raise ValueError(
            "Logical coding inconsistency: when "
            "geography_present=no, evidence_status "
            "must be none, direction must be none, "
            "and decision_relevance must be no. "
            f"Examples: "
            f"{bad_no_geo['coding_id'].tolist()[:10]}"
        )

    unsupported = frame[
        "evidence_status"
    ].isin(
        {
            "unsupported_general_inference",
            "unsupplied_specific_geographic_claim",
        }
    )

    bad_unsupported = frame[
        unsupported
        & ~frame[
            "geography_present"
        ].eq("yes")
    ]

    if not bad_unsupported.empty:
        raise ValueError(
            "Unsupported geographic reasoning "
            "requires geography_present=yes. "
            f"Examples: "
            f"{bad_unsupported['coding_id'].tolist()[:10]}"
        )


def apply_adjudication(
    researcher: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    final = researcher.copy()

    if not path.exists():
        return final

    adjudication = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    if adjudication.empty:
        return final

    required = [
        "coding_id",
        *[
            f"adjudicated_{dimension}"
            for dimension
            in DIMENSIONS
        ],
    ]

    missing = [
        column
        for column in required
        if column not in adjudication.columns
    ]

    if missing:
        raise ValueError(
            "Adjudication file missing columns: "
            f"{missing}"
        )

    if "any_disagreement" in adjudication.columns:
        flag = normalize(
            adjudication[
                "any_disagreement"
            ]
        )

        adjudication = adjudication[
            flag.isin(
                {
                    "true",
                    "1",
                    "yes",
                }
            )
        ].copy()

    if adjudication.empty:
        return final

    if adjudication["coding_id"].duplicated().any():
        raise ValueError(
            "Adjudication file contains duplicate "
            "coding_id values."
        )

    researcher_ids = set(
        final["coding_id"]
    )

    unknown_ids = sorted(
        set(
            adjudication[
                "coding_id"
            ].astype(str)
        )
        - researcher_ids
    )

    if unknown_ids:
        raise ValueError(
            "Adjudication contains coding IDs "
            "not found in researcher file: "
            f"{unknown_ids[:10]}"
        )

    final = final.set_index(
        "coding_id"
    )

    for _, row in adjudication.iterrows():
        coding_id = str(
            row["coding_id"]
        ).strip()

        for dimension, allowed in DIMENSIONS.items():
            column = (
                f"adjudicated_{dimension}"
            )

            value = str(
                row[column]
            ).strip().lower()

            if value == "":
                raise ValueError(
                    "Adjudication is incomplete for "
                    f"{coding_id}: {column} is blank."
                )

            if value not in allowed:
                raise ValueError(
                    f"Invalid adjudicated value "
                    f"for {coding_id}, {dimension}: "
                    f"{value}. Allowed: "
                    f"{sorted(allowed)}"
                )

            final.loc[
                coding_id,
                dimension,
            ] = value

    return final.reset_index()


def load_private_key(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Private coding key not found: {path}"
        )

    key = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    required = [
        "coding_id",
        "run_id",
        "model_id",
        "scenario_id",
        "strength",
        "condition",
        "country",
        "repetition",
    ]

    missing = [
        column
        for column in required
        if column not in key.columns
    ]

    if missing:
        raise ValueError(
            "Private coding key missing required "
            f"columns: {missing}"
        )

    key["coding_id"] = (
        key["coding_id"]
        .astype(str)
        .str.strip()
    )

    if key["coding_id"].duplicated().any():
        raise ValueError(
            "Private coding key contains duplicate "
            "coding_id values."
        )

    invalid_conditions = sorted(
        set(
            key["condition"]
            .astype(str)
            .str.strip()
        )
        - VALID_CONDITIONS
    )

    if invalid_conditions:
        raise ValueError(
            "Unexpected condition values in "
            f"private key: {invalid_conditions}"
        )

    return key[
        required
    ].copy()


def derive_variables(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result[
        "unsupported_geography"
    ] = (
        result["evidence_status"].isin(
            {
                "unsupported_general_inference",
                "unsupplied_specific_geographic_claim",
            }
        )
    ).astype(int)

    result[
        "unsupported_negative"
    ] = (
        result[
            "unsupported_geography"
        ].eq(1)
        & result["direction"].isin(
            {
                "negative",
                "mixed",
            }
        )
    ).astype(int)

    result[
        "unsupported_positive"
    ] = (
        result[
            "unsupported_geography"
        ].eq(1)
        & result["direction"].isin(
            {
                "positive",
                "mixed",
            }
        )
    ).astype(int)

    result[
        "geography_decision_relevant"
    ] = (
        result[
            "geography_present"
        ].eq("yes")
        & result[
            "decision_relevance"
        ].eq("yes")
    ).astype(int)

    return result


def summarize_rates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for condition, group in frame.groupby(
        "condition",
        sort=True,
    ):
        for variable in DERIVED_VARIABLES:
            numerator = int(
                group[variable].sum()
            )

            denominator = int(
                len(group)
            )

            percent = (
                100.0
                * numerator
                / denominator
                if denominator
                else float("nan")
            )

            rows.append(
                {
                    "condition": condition,
                    "measure": variable,
                    "numerator": numerator,
                    "denominator": denominator,
                    "percent": percent,
                }
            )

    return pd.DataFrame(
        rows
    )


def mitigation_comparison(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for condition in [
        "labelled",
        "mitigation",
    ]:
        subset = frame[
            frame["condition"].eq(
                condition
            )
        ]

        numerator = int(
            subset[
                "unsupported_geography"
            ].sum()
        )

        denominator = int(
            len(subset)
        )

        percent = (
            100.0
            * numerator
            / denominator
            if denominator
            else float("nan")
        )

        rows.append(
            {
                "condition": condition,
                "unsupported_numerator": (
                    numerator
                ),
                "denominator": denominator,
                "unsupported_percent": (
                    percent
                ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    labelled_rate = float(
        comparison.loc[
            comparison[
                "condition"
            ].eq("labelled"),
            "unsupported_percent",
        ].iloc[0]
    )

    mitigation_rate = float(
        comparison.loc[
            comparison[
                "condition"
            ].eq("mitigation"),
            "unsupported_percent",
        ].iloc[0]
    )

    comparison[
        "mitigation_minus_labelled_pp"
    ] = float("nan")

    comparison.loc[
        comparison[
            "condition"
        ].eq("mitigation"),
        "mitigation_minus_labelled_pp",
    ] = (
        mitigation_rate
        - labelled_rate
    )

    return comparison


def main() -> None:
    args = parse_args()

    researcher = load_researcher(
        args.researcher
    )

    final_codes = apply_adjudication(
        researcher,
        args.adjudication,
    )

    validate_consistency(
        final_codes
    )

    private_key = load_private_key(
        args.private_key
    )

    merged = final_codes.merge(
        private_key,
        on="coding_id",
        how="left",
        validate="one_to_one",
    )

    missing_metadata = merged[
        merged["run_id"].isna()
        | merged["run_id"].eq("")
    ]

    if not missing_metadata.empty:
        raise ValueError(
            "Some coding IDs are missing from "
            "the private metadata key: "
            f"{missing_metadata['coding_id'].tolist()[:10]}"
        )

    merged = derive_variables(
        merged
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    item_output = (
        args.out_dir
        / "qualitative_item_level.csv"
    )

    rate_output = (
        args.out_dir
        / "qualitative_rates_by_condition.csv"
    )

    mitigation_output = (
        args.out_dir
        / "unsupported_geography_labelled_vs_mitigation.csv"
    )

    merged[
        [
            "coding_id",
            "run_id",
            "scenario_id",
            "model_id",
            "strength",
            "condition",
            "country",
            "repetition",
            *DIMENSIONS.keys(),
            *DERIVED_VARIABLES,
        ]
    ].to_csv(
        item_output,
        index=False,
    )

    summary = summarize_rates(
        merged
    )

    summary.to_csv(
        rate_output,
        index=False,
    )

    comparison = mitigation_comparison(
        merged
    )

    comparison.to_csv(
        mitigation_output,
        index=False,
    )

    print("=" * 72)
    print(
        "QUALITATIVE ANALYSIS COMPLETE"
    )
    print("=" * 72)
    print(
        f"Eligible coded items: {len(merged)}"
    )
    print()
    print(
        summary.to_string(
            index=False
        )
    )
    print()
    print(
        "Labelled vs mitigation "
        "unsupported-geography comparison:"
    )
    print(
        comparison.to_string(
            index=False
        )
    )
    print()
    print(
        "No significance test was added "
        "to the qualitative rate comparison."
    )
    print(
        f"Outputs written to: "
        f"{args.out_dir}"
    )


if __name__ == "__main__":
    main()
