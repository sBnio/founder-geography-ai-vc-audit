from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DIMENSIONS = {
    "geography_present": [
        "yes",
        "no",
    ],
    "evidence_status": [
        "none",
        "neutral_restatement",
        "supported",
        "unsupported_general_inference",
        "unsupplied_specific_geographic_claim",
    ],
    "direction": [
        "none",
        "neutral",
        "negative",
        "positive",
        "mixed",
    ],
    "decision_relevance": [
        "no",
        "yes",
        "unclear",
    ],
    "coder_confidence": [
        "high",
        "medium",
        "low",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate pre-adjudication agreement between "
            "the researcher and independent second coder."
        )
    )

    parser.add_argument(
        "--researcher",
        type=Path,
        default=ROOT / "data/coding/researcher_blinded_full.csv",
        help="Completed researcher coding CSV.",
    )

    parser.add_argument(
        "--second-coder",
        type=Path,
        default=ROOT / "data/coding/second_coder_blinded_sample.csv",
        help="Completed independent second-coder CSV.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/coding_agreement",
        help="Directory for agreement outputs.",
    )

    return parser.parse_args()


def load_codes(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file does not exist: {path}"
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
            f"{label} file is missing required columns: "
            f"{missing}"
        )

    frame = frame[required].copy()

    frame["coding_id"] = (
        frame["coding_id"]
        .astype(str)
        .str.strip()
    )

    if frame["coding_id"].eq("").any():
        raise ValueError(
            f"{label} file contains blank coding_id values."
        )

    duplicated = (
        frame.loc[
            frame["coding_id"].duplicated(
                keep=False
            ),
            "coding_id",
        ]
        .tolist()
    )

    if duplicated:
        raise ValueError(
            f"{label} file contains duplicate coding_id "
            f"values: {duplicated[:10]}"
        )

    for column in DIMENSIONS:
        frame[column] = (
            frame[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return frame


def nominal_krippendorff_alpha(
    researcher: list[str],
    second_coder: list[str],
) -> float:
    """
    Nominal Krippendorff alpha for two complete coders.

    Observed disagreement is the proportion of matched
    units on which the two coders disagree.

    Expected disagreement is calculated from the pooled
    category frequencies using the finite-sample
    coincidence formulation.
    """

    if len(researcher) != len(second_coder):
        raise ValueError(
            "Coder vectors must have equal length."
        )

    n_units = len(researcher)

    if n_units == 0:
        return math.nan

    disagreements = sum(
        a != b
        for a, b in zip(
            researcher,
            second_coder,
            strict=True,
        )
    )

    observed_disagreement = (
        disagreements / n_units
    )

    pooled = [
        *researcher,
        *second_coder,
    ]

    n_values = len(pooled)

    if n_values < 2:
        return math.nan

    counts = Counter(pooled)

    expected_agreement = (
        sum(
            count * (count - 1)
            for count in counts.values()
        )
        /
        (
            n_values
            * (n_values - 1)
        )
    )

    expected_disagreement = (
        1.0 - expected_agreement
    )

    if math.isclose(
        expected_disagreement,
        0.0,
        abs_tol=1e-15,
    ):
        return math.nan

    return (
        1.0
        - (
            observed_disagreement
            / expected_disagreement
        )
    )


def validate_values(
    matched: pd.DataFrame,
) -> None:
    for dimension, allowed in DIMENSIONS.items():
        for coder in [
            "researcher",
            "second_coder",
        ]:
            column = (
                f"{dimension}_{coder}"
            )

            blanks = matched[
                matched[column].eq("")
            ]

            if not blanks.empty:
                ids = (
                    blanks["coding_id"]
                    .astype(str)
                    .tolist()
                )

                raise ValueError(
                    f"Blank {dimension} values in "
                    f"{coder} coding for IDs: "
                    f"{ids[:10]}"
                )

            invalid = matched[
                ~matched[column].isin(
                    allowed
                )
            ]

            if not invalid.empty:
                examples = (
                    invalid[
                        [
                            "coding_id",
                            column,
                        ]
                    ]
                    .head(10)
                    .to_dict(
                        orient="records"
                    )
                )

                raise ValueError(
                    f"Invalid values for "
                    f"{column}. "
                    f"Allowed: {allowed}. "
                    f"Examples: {examples}"
                )


def validate_logical_consistency(
    matched: pd.DataFrame,
) -> None:
    """
    Enforce the codebook's hard consistency rule:

    geography_present=no ->
        evidence_status=none
        direction=none
        decision_relevance=no

    Other codebook relationships described as
    'normally' or 'should normally' are not silently
    enforced here.
    """

    for coder in [
        "researcher",
        "second_coder",
    ]:
        geography = (
            f"geography_present_{coder}"
        )
        evidence = (
            f"evidence_status_{coder}"
        )
        direction = (
            f"direction_{coder}"
        )
        relevance = (
            f"decision_relevance_{coder}"
        )

        bad = matched[
            matched[geography].eq("no")
            & (
                ~matched[evidence].eq(
                    "none"
                )
                | ~matched[direction].eq(
                    "none"
                )
                | ~matched[relevance].eq(
                    "no"
                )
            )
        ]

        if not bad.empty:
            examples = (
                bad[
                    [
                        "coding_id",
                        geography,
                        evidence,
                        direction,
                        relevance,
                    ]
                ]
                .head(10)
                .to_dict(
                    orient="records"
                )
            )

            raise ValueError(
                "Codebook consistency failure "
                f"for {coder}: when "
                "geography_present=no, "
                "evidence_status must be none, "
                "direction must be none, and "
                "decision_relevance must be no. "
                f"Examples: {examples}"
            )


def confusion_matrix(
    researcher: pd.Series,
    second_coder: pd.Series,
    categories: list[str],
    dimension: str,
) -> pd.DataFrame:
    researcher_cat = pd.Categorical(
        researcher,
        categories=categories,
    )

    second_cat = pd.Categorical(
        second_coder,
        categories=categories,
    )

    matrix = pd.crosstab(
        researcher_cat,
        second_cat,
        dropna=False,
    )

    matrix = matrix.reindex(
        index=categories,
        columns=categories,
        fill_value=0,
    )

    matrix.index.name = (
        f"researcher_{dimension}"
    )

    matrix.columns.name = (
        f"second_coder_{dimension}"
    )

    return matrix.astype(int)


def main() -> None:
    args = parse_args()

    researcher = load_codes(
        args.researcher,
        "Researcher",
    )

    second_coder = load_codes(
        args.second_coder,
        "Second coder",
    )

    missing_from_researcher = sorted(
        set(
            second_coder[
                "coding_id"
            ]
        )
        - set(
            researcher[
                "coding_id"
            ]
        )
    )

    if missing_from_researcher:
        raise ValueError(
            "Second-coder coding IDs are not "
            "present in the researcher file: "
            f"{missing_from_researcher[:10]}"
        )

    matched = researcher.merge(
        second_coder,
        on="coding_id",
        how="inner",
        suffixes=(
            "_researcher",
            "_second_coder",
        ),
        validate="one_to_one",
    )

    if matched.empty:
        raise ValueError(
            "No double-coded items matched "
            "by coding_id."
        )

    if len(matched) != len(
        second_coder
    ):
        raise ValueError(
            "Not every second-coder item "
            "matched a researcher item."
        )

    validate_values(
        matched
    )

    validate_logical_consistency(
        matched
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for (
        dimension,
        categories,
    ) in DIMENSIONS.items():

        researcher_col = (
            f"{dimension}_researcher"
        )

        second_col = (
            f"{dimension}_second_coder"
        )

        r_values = (
            matched[
                researcher_col
            ]
            .astype(str)
            .tolist()
        )

        s_values = (
            matched[
                second_col
            ]
            .astype(str)
            .tolist()
        )

        agreement_count = sum(
            a == b
            for a, b in zip(
                r_values,
                s_values,
                strict=True,
            )
        )

        disagreement_count = (
            len(matched)
            - agreement_count
        )

        percent_agreement = (
            100.0
            * agreement_count
            / len(matched)
        )

        alpha = (
            nominal_krippendorff_alpha(
                r_values,
                s_values,
            )
        )

        summary_rows.append(
            {
                "dimension": dimension,
                "n_double_coded": len(
                    matched
                ),
                "agreement_count": (
                    agreement_count
                ),
                "disagreement_count": (
                    disagreement_count
                ),
                "percent_agreement": (
                    percent_agreement
                ),
                "krippendorff_alpha_nominal": (
                    alpha
                ),
                "alpha_note": (
                    ""
                    if not math.isnan(
                        alpha
                    )
                    else (
                        "undefined because "
                        "expected disagreement "
                        "is zero or insufficient "
                        "variation exists"
                    )
                ),
            }
        )

        matrix = confusion_matrix(
            matched[
                researcher_col
            ],
            matched[
                second_col
            ],
            categories,
            dimension,
        )

        matrix.to_csv(
            args.out_dir
            / (
                "confusion_"
                f"{dimension}.csv"
            )
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        args.out_dir
        / "agreement_summary.csv",
        index=False,
    )

    print(
        "=" * 72
    )
    print(
        "PRE-ADJUDICATION CODING AGREEMENT"
    )
    print(
        "=" * 72
    )
    print(
        f"Researcher file: {args.researcher}"
    )
    print(
        f"Second-coder file: {args.second_coder}"
    )
    print(
        f"Double-coded items: {len(matched)}"
    )
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Outputs written to: "
        f"{args.out_dir}"
    )
    print(
        "Original coder files were read only "
        "and were not modified."
    )


if __name__ == "__main__":
    main()
