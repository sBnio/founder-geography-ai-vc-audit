from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DIMENSIONS = [
    "geography_present",
    "evidence_status",
    "direction",
    "decision_relevance",
    "coder_confidence",
]

TEXT_COLUMNS = [
    "startup_case",
    "decision_factors_json",
    "assumptions_or_inferences_json",
    "missing_information_json",
    "one_sentence_rationale",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a separate adjudication sheet "
            "for qualitative coding disagreements. "
            "Original coder files are read only."
        )
    )

    parser.add_argument(
        "--researcher",
        type=Path,
        default=(
            ROOT
            / "data/coding/researcher_blinded_full.csv"
        ),
    )

    parser.add_argument(
        "--second-coder",
        type=Path,
        default=(
            ROOT
            / "data/coding/second_coder_blinded_sample.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "data/coding/adjudication_sheet.csv"
        ),
    )

    return parser.parse_args()


def read_file(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found: {path}"
        )

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    required = [
        "coding_id",
        *DIMENSIONS,
    ]

    missing = [
        col
        for col in required
        if col not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{label} file missing columns: "
            f"{missing}"
        )

    frame["coding_id"] = (
        frame["coding_id"]
        .astype(str)
        .str.strip()
    )

    if frame["coding_id"].eq("").any():
        raise ValueError(
            f"{label} contains blank coding_id."
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
            f"{label} contains duplicate "
            f"coding_id values: "
            f"{duplicates[:10]}"
        )

    return frame


def main() -> None:
    args = parse_args()

    researcher = read_file(
        args.researcher,
        "Researcher",
    )

    second = read_file(
        args.second_coder,
        "Second coder",
    )

    researcher_ids = set(
        researcher["coding_id"]
    )

    missing_ids = sorted(
        set(second["coding_id"])
        - researcher_ids
    )

    if missing_ids:
        raise ValueError(
            "Second-coder IDs missing from "
            "researcher file: "
            f"{missing_ids[:10]}"
        )

    researcher_keep = [
        "coding_id",
        *[
            col
            for col in TEXT_COLUMNS
            if col in researcher.columns
        ],
        *DIMENSIONS,
    ]

    merged = (
        researcher[
            researcher_keep
        ]
        .merge(
            second[
                [
                    "coding_id",
                    *DIMENSIONS,
                ]
            ],
            on="coding_id",
            how="inner",
            suffixes=(
                "_researcher",
                "_second_coder",
            ),
            validate="one_to_one",
        )
    )

    if len(merged) != len(second):
        raise ValueError(
            "Not every second-coder row "
            "matched exactly one researcher row."
        )

    disagreement_flags = []

    for dimension in DIMENSIONS:
        r_col = (
            f"{dimension}_researcher"
        )
        s_col = (
            f"{dimension}_second_coder"
        )

        flag_col = (
            f"disagree_{dimension}"
        )

        merged[flag_col] = (
            merged[r_col]
            .astype(str)
            .str.strip()
            .str.lower()
            !=
            merged[s_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        disagreement_flags.append(
            flag_col
        )

    merged[
        "any_disagreement"
    ] = merged[
        disagreement_flags
    ].any(
        axis=1
    )

    disagreements = (
        merged.loc[
            merged[
                "any_disagreement"
            ]
        ]
        .copy()
    )

    # Blank adjudication fields:
    # a human fills these AFTER agreement
    # has already been calculated.
    for dimension in DIMENSIONS:
        disagreements[
            f"adjudicated_{dimension}"
        ] = ""

    disagreements[
        "adjudication_notes"
    ] = ""

    disagreements[
        "adjudicated_by"
    ] = ""

    disagreements[
        "adjudication_date_utc"
    ] = ""

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output.exists():
        raise FileExistsError(
            "Refusing to overwrite existing "
            f"adjudication file: {args.output}"
        )

    disagreements.to_csv(
        args.output,
        index=False,
    )

    print(
        "=" * 72
    )
    print(
        "QUALITATIVE ADJUDICATION SHEET"
    )
    print(
        "=" * 72
    )
    print(
        f"Double-coded items: "
        f"{len(merged)}"
    )
    print(
        f"Items with >=1 disagreement: "
        f"{len(disagreements)}"
    )
    print(
        f"Output: {args.output}"
    )
    print(
        "Original coder files were "
        "read only and not modified."
    )


if __name__ == "__main__":
    main()
