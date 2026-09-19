from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TEXT_COLUMNS = [
    "decision_factors_json",
    "assumptions_or_inferences_json",
    "missing_information_json",
    "one_sentence_rationale",
]

STRATIFICATION_COLUMNS = [
    "condition",
    "model_id",
    "country",
    "strength",
]

PRIVATE_METADATA_COLUMNS = [
    "run_id",
    "provider",
    "model_id",
    "actual_model_returned",
    "scenario_id",
    "strength",
    "condition",
    "country",
    "repetition",
    "decision",
    "investment_score_0_100",
    "recommended_investment_usd",
    "recommended_pre_money_valuation_usd",
    "probability_of_closing_stated_round_within_12_months_percent",
    "financial_risk_score_0_100",
    "confidence_0_100",
]

CODING_COLUMNS = [
    "coding_id",
    "startup_case",
    "decision_factors_json",
    "assumptions_or_inferences_json",
    "missing_information_json",
    "one_sentence_rationale",
    "geography_present",
    "evidence_status",
    "direction",
    "decision_relevance",
    "coder_confidence",
    "coder_notes",
]

LOCATION_TOKEN = "[DISCLOSED LOCATION]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create v3 blinded qualitative-coding files "
            "and deterministic second-coder sample."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/processed/responses.csv",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/coding",
    )

    parser.add_argument(
        "--sample-n",
        type=int,
        default=72,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260720,
    )

    return parser.parse_args()


def stable_random_state(
    seed: int,
    *parts: str,
) -> int:
    text = "|".join(
        [str(seed), *map(str, parts)]
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16)


def target_second_coder_sample_size(
    eligible_n: int,
    full_sample_n: int = 72,
) -> int:
    """72 at full 360; >=20% for a smaller eligible universe."""

    if eligible_n < 1:
        return 0

    if eligible_n >= 360:
        return min(
            eligible_n,
            full_sample_n,
        )

    return min(
        eligible_n,
        max(
            1,
            math.ceil(
                eligible_n * 0.20
            ),
        ),
    )


def mask_location_terms(value: object) -> str:
    """Mask direct treatment names/demonyms, not other geography."""

    if pd.isna(value):
        return ""

    text = str(value)

    # Long names first.
    name_patterns = [
        r"\bthe\s+United\s+States\b",
        r"\bUnited\s+States\s+of\s+America\b",
        r"\bUnited\s+States\b",
        r"\bKyrgyzstan\b",
        r"\bKyrgyzstani\b",
        r"\bKyrgyz\b",
        r"\bAmericans\b",
        r"\bAmerican\b",
    ]

    for pattern in name_patterns:
        text = re.sub(
            pattern,
            LOCATION_TOKEN,
            text,
            flags=re.IGNORECASE,
        )

    # Keep abbreviations case-sensitive so ordinary
    # pronoun "us" is never masked accidentally.
    abbreviation_patterns = [
        r"(?<!\w)U\.S\.A\.?(?!\w)",
        r"(?<!\w)USA(?!\w)",
        r"(?<!\w)U\.S\.?(?!\w)",
        r"(?<!\w)US(?!\w)",
    ]

    for pattern in abbreviation_patterns:
        text = re.sub(
            pattern,
            LOCATION_TOKEN,
            text,
        )

    return text


def load_case_map() -> dict[str, dict]:
    path = ROOT / "scenarios/final_cases.json"

    cases = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return {
        str(case["id"]): case
        for case in cases
    }


def blinded_case_text(
    row: pd.Series,
    case_map: dict[str, dict],
) -> str:
    """Reconstruct the supplied fictional case with treatment masked."""

    scenario_id = str(
        row["scenario_id"]
    )

    if scenario_id not in case_map:
        raise ValueError(
            f"Unknown scenario_id: {scenario_id}"
        )

    template = str(
        case_map[
            scenario_id
        ]["pitch_template"]
    ).strip()

    if template.count("[COUNTRY]") != 1:
        raise ValueError(
            f"{scenario_id}: expected exactly one "
            "[COUNTRY] placeholder"
        )

    condition = str(
        row["condition"]
    )

    if condition == "anonymous":
        sentences = re.split(
            r"(?<=[.!?])\s+",
            template,
        )

        matching = [
            sentence
            for sentence in sentences
            if "[COUNTRY]" in sentence
        ]

        if len(matching) != 1:
            raise ValueError(
                f"{scenario_id}: could not identify "
                "exactly one location sentence"
            )

        sentences = [
            sentence
            for sentence in sentences
            if "[COUNTRY]" not in sentence
        ]

        return " ".join(
            sentences
        ).strip()

    if condition in {
        "labelled",
        "mitigation",
    }:
        return template.replace(
            "[COUNTRY]",
            LOCATION_TOKEN,
        )

    raise ValueError(
        f"Unexpected condition: {condition}"
    )


def allocate_stratified_counts(
    strata: pd.DataFrame,
    target: int,
) -> pd.DataFrame:
    """Allocate a deterministic sample across realized strata."""

    result = strata.copy().reset_index(
        drop=True
    )

    if target < 0:
        raise ValueError(
            "target must be nonnegative"
        )

    total_available = int(
        result["eligible_n"].sum()
    )

    if target > total_available:
        raise ValueError(
            "target exceeds eligible universe"
        )

    result["selected_n"] = 0

    if target == 0 or result.empty:
        return result

    n_strata = len(result)

    # When feasible, guarantee representation of every
    # realized condition x model x country x strength stratum.
    if target >= n_strata:
        result["selected_n"] = 1
        remaining = target - n_strata

        capacity = (
            result["eligible_n"]
            - result["selected_n"]
        )
    else:
        remaining = target
        capacity = result[
            "eligible_n"
        ].copy()

    if remaining == 0:
        return result

    capacity_total = int(
        capacity.sum()
    )

    if capacity_total <= 0:
        return result

    exact = (
        capacity.astype(float)
        / capacity_total
        * remaining
    )

    base = (
        exact.apply(math.floor)
        .astype(int)
    )

    base = base.clip(
        upper=capacity
    )

    result["selected_n"] += base

    remainder_to_allocate = (
        target
        - int(
            result[
                "selected_n"
            ].sum()
        )
    )

    fractional = exact - base

    tie_keys = result[
        STRATIFICATION_COLUMNS
    ].astype(str).agg(
        "|".join,
        axis=1,
    )

    order = sorted(
        range(len(result)),
        key=lambda i: (
            -float(
                fractional.iloc[i]
            ),
            tie_keys.iloc[i],
        ),
    )

    while remainder_to_allocate > 0:
        changed = False

        for i in order:
            if (
                result.loc[
                    i,
                    "selected_n",
                ]
                >= result.loc[
                    i,
                    "eligible_n",
                ]
            ):
                continue

            result.loc[
                i,
                "selected_n",
            ] += 1

            remainder_to_allocate -= 1
            changed = True

            if remainder_to_allocate == 0:
                break

        if not changed:
            raise RuntimeError(
                "Unable to complete "
                "stratified allocation"
            )

    if int(
        result["selected_n"].sum()
    ) != target:
        raise RuntimeError(
            "Stratified allocation has "
            "incorrect final size"
        )

    return result


def make_coder_sheet(
    rows: pd.DataFrame,
    case_map: dict[str, dict],
) -> pd.DataFrame:
    """Create a treatment/model/outcome-blinded coding sheet."""

    output = pd.DataFrame()

    output["coding_id"] = (
        rows["coding_id"].astype(str)
    )

    output["startup_case"] = [
        mask_location_terms(
            blinded_case_text(
                row,
                case_map,
            )
        )
        for _, row in rows.iterrows()
    ]

    for column in TEXT_COLUMNS:
        output[column] = (
            rows[column]
            .fillna("")
            .map(mask_location_terms)
        )

    # Human coders fill these manually.
    output["geography_present"] = ""
    output["evidence_status"] = ""
    output["direction"] = ""
    output["decision_relevance"] = ""
    output["coder_confidence"] = ""
    output["coder_notes"] = ""

    return output[
        CODING_COLUMNS
    ]


def direct_treatment_leaks(
    frame: pd.DataFrame,
) -> list[str]:
    """Return coder-facing cells containing direct treatment names."""

    leak_patterns = [
        r"\bKyrgyzstan\b",
        r"\bKyrgyzstani\b",
        r"\bKyrgyz\b",
        r"\bUnited\s+States\b",
        r"\bAmerican\b",
        r"\bAmericans\b",
        r"(?<!\w)U\.S\.A\.?(?!\w)",
        r"(?<!\w)USA(?!\w)",
        r"(?<!\w)U\.S\.?(?!\w)",
        r"(?<!\w)US(?!\w)",
    ]

    leaks = []

    text_columns = [
        "startup_case",
        *TEXT_COLUMNS,
    ]

    for column in text_columns:
        for row_index, value in frame[
            column
        ].items():
            text = str(value)

            for pattern in leak_patterns:
                flags = (
                    0
                    if pattern.startswith(
                        "(?<!"
                    )
                    else re.IGNORECASE
                )

                if re.search(
                    pattern,
                    text,
                    flags=flags,
                ):
                    leaks.append(
                        f"{column}:{row_index}"
                    )
                    break

    return leaks


def ensure_outputs_do_not_exist(
    paths: list[Path],
) -> None:
    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if existing:
        joined = "\n".join(
            str(path)
            for path in existing
        )

        raise SystemExit(
            "ERROR: Refusing to overwrite existing "
            "coder files. Preserve originals.\n"
            f"{joined}"
        )


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"ERROR: Input file not found: {args.input}"
        )

    if args.sample_n < 1:
        raise SystemExit(
            "ERROR: sample-n must be positive."
        )

    output_paths = {
        "researcher":
            args.out_dir
            / "researcher_blinded_full.csv",
        "second":
            args.out_dir
            / "second_coder_blinded_sample.csv",
        "key":
            args.out_dir
            / "PRIVATE_coding_key.csv",
        "allocation":
            args.out_dir
            / "PRIVATE_sampling_allocation.csv",
    }

    ensure_outputs_do_not_exist(
        list(
            output_paths.values()
        )
    )

    df = pd.read_csv(
        args.input
    )

    required = {
        "run_id",
        "provider",
        "model_id",
        "scenario_id",
        "strength",
        "condition",
        "country",
        "status",
        *TEXT_COLUMNS,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise SystemExit(
            "ERROR: Missing required columns: "
            f"{sorted(missing)}"
        )

    eligible = df[
        df["status"]
        .astype(str)
        .str.lower()
        == "success"
    ].copy()

    if eligible.empty:
        raise SystemExit(
            "ERROR: No eligible successful outputs."
        )

    if (
        eligible["run_id"].nunique()
        != len(eligible)
    ):
        raise SystemExit(
            "ERROR: Duplicate run IDs "
            "in eligible universe."
        )

    for column in TEXT_COLUMNS:
        eligible[column] = (
            eligible[column]
            .fillna("")
            .astype(str)
        )

    case_map = load_case_map()

    unknown_cases = set(
        eligible[
            "scenario_id"
        ].astype(str)
    ).difference(
        case_map
    )

    if unknown_cases:
        raise SystemExit(
            "ERROR: Unknown scenarios in input: "
            f"{sorted(unknown_cases)}"
        )

    # --------------------------------------------------------
    # Researcher full universe: deterministic blinded order.
    # --------------------------------------------------------

    researcher_rows = (
        eligible.sort_values(
            "run_id"
        )
        .sample(
            frac=1,
            random_state=stable_random_state(
                args.seed,
                "researcher-full-order",
            ),
        )
        .reset_index(
            drop=True
        )
    )

    researcher_rows.insert(
        0,
        "coding_id",
        [
            f"C{i:04d}"
            for i in range(
                1,
                len(
                    researcher_rows
                ) + 1,
            )
        ],
    )

    coding_id_by_run = dict(
        zip(
            researcher_rows[
                "run_id"
            ].astype(str),
            researcher_rows[
                "coding_id"
            ].astype(str),
        )
    )

    researcher_sheet = (
        make_coder_sheet(
            researcher_rows,
            case_map,
        )
    )

    # --------------------------------------------------------
    # Deterministic second-coder stratified subset.
    # --------------------------------------------------------

    target_n = (
        target_second_coder_sample_size(
            len(eligible),
            args.sample_n,
        )
    )

    strata = (
        eligible.groupby(
            STRATIFICATION_COLUMNS,
            dropna=False,
        )
        .size()
        .rename(
            "eligible_n"
        )
        .reset_index()
        .sort_values(
            STRATIFICATION_COLUMNS
        )
        .reset_index(
            drop=True
        )
    )

    allocation = (
        allocate_stratified_counts(
            strata,
            target_n,
        )
    )

    selected_parts = []

    for _, stratum in allocation.iterrows():
        quota = int(
            stratum[
                "selected_n"
            ]
        )

        if quota == 0:
            continue

        mask = pd.Series(
            True,
            index=eligible.index,
        )

        for column in STRATIFICATION_COLUMNS:
            mask &= (
                eligible[column].astype(str)
                == str(
                    stratum[
                        column
                    ]
                )
            )

        group = eligible[
            mask
        ]

        if len(group) < quota:
            raise RuntimeError(
                "Allocated quota exceeds "
                "stratum availability"
            )

        random_state = (
            stable_random_state(
                args.seed,
                *[
                    str(
                        stratum[
                            column
                        ]
                    )
                    for column
                    in STRATIFICATION_COLUMNS
                ],
            )
        )

        selected_parts.append(
            group.sample(
                n=quota,
                replace=False,
                random_state=random_state,
            )
        )

    second_rows = pd.concat(
        selected_parts,
        ignore_index=False,
    )

    if len(second_rows) != target_n:
        raise RuntimeError(
            "Incorrect second-coder sample size: "
            f"{len(second_rows)} != {target_n}"
        )

    if (
        second_rows[
            "run_id"
        ].nunique()
        != len(second_rows)
    ):
        raise RuntimeError(
            "Duplicate run IDs in "
            "second-coder sample"
        )

    second_rows = (
        second_rows.assign(
            coding_id=lambda frame:
                frame["run_id"]
                .astype(str)
                .map(
                    coding_id_by_run
                )
        )
        .sample(
            frac=1,
            random_state=stable_random_state(
                args.seed,
                "second-coder-order",
            ),
        )
        .reset_index(
            drop=True
        )
    )

    second_sheet = (
        make_coder_sheet(
            second_rows,
            case_map,
        )
    )

    # --------------------------------------------------------
    # Leakage checks before writing coder-facing files.
    # --------------------------------------------------------

    researcher_leaks = (
        direct_treatment_leaks(
            researcher_sheet
        )
    )

    second_leaks = (
        direct_treatment_leaks(
            second_sheet
        )
    )

    if researcher_leaks:
        raise RuntimeError(
            "Direct treatment-name leakage "
            "in researcher sheet: "
            f"{researcher_leaks[:10]}"
        )

    if second_leaks:
        raise RuntimeError(
            "Direct treatment-name leakage "
            "in second-coder sheet: "
            f"{second_leaks[:10]}"
        )

    # --------------------------------------------------------
    # Private key for all eligible outputs.
    # --------------------------------------------------------

    private_key = (
        researcher_rows[
            [
                "coding_id",
                *[
                    column
                    for column
                    in PRIVATE_METADATA_COLUMNS
                    if column
                    in researcher_rows.columns
                ],
            ]
        ]
        .copy()
    )

    selected_run_ids = set(
        second_rows[
            "run_id"
        ].astype(str)
    )

    private_key[
        "selected_for_second_coder"
    ] = (
        private_key[
            "run_id"
        ]
        .astype(str)
        .isin(
            selected_run_ids
        )
    )

    second_order = {
        str(run_id): int(index + 1)
        for index, run_id
        in enumerate(
            second_rows[
                "run_id"
            ].astype(str)
        )
    }

    private_key[
        "second_coder_order"
    ] = (
        private_key[
            "run_id"
        ]
        .astype(str)
        .map(
            second_order
        )
    )

    allocation = allocation.copy()

    allocation[
        "target_total_sample_n"
    ] = target_n

    allocation[
        "eligible_universe_n"
    ] = len(eligible)

    allocation[
        "seed"
    ] = args.seed

    # --------------------------------------------------------
    # Write once. Original coder files must be preserved.
    # --------------------------------------------------------

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    researcher_sheet.to_csv(
        output_paths[
            "researcher"
        ],
        index=False,
    )

    second_sheet.to_csv(
        output_paths[
            "second"
        ],
        index=False,
    )

    private_key.to_csv(
        output_paths[
            "key"
        ],
        index=False,
    )

    allocation.to_csv(
        output_paths[
            "allocation"
        ],
        index=False,
    )

    for private_path in [
        output_paths["key"],
        output_paths["allocation"],
    ]:
        try:
            os.chmod(
                private_path,
                0o600,
            )
        except OSError:
            pass

    print("=" * 76)
    print("V3 BLINDED QUALITATIVE CODING FILES")
    print("=" * 76)

    print(
        f"Eligible qualitative universe: "
        f"{len(eligible)}"
    )

    print(
        f"Researcher blinded rows:       "
        f"{len(researcher_sheet)}"
    )

    print(
        f"Second-coder sample rows:      "
        f"{len(second_sheet)}"
    )

    print(
        f"Realized strata:               "
        f"{len(allocation)}"
    )

    print(
        "Direct treatment-name leaks:  0"
    )

    print(
        "\nSecond-coder stratification:"
    )

    print(
        allocation[
            [
                *STRATIFICATION_COLUMNS,
                "eligible_n",
                "selected_n",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nCreated:")
    for path in output_paths.values():
        print(path)

    print(
        "\nIMPORTANT: second coder receives only "
        "second_coder_blinded_sample.csv plus "
        "the approved codebook/training materials."
    )

    print(
        "Never send PRIVATE_coding_key.csv or "
        "PRIVATE_sampling_allocation.csv."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
