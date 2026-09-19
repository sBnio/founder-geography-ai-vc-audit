import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.analyze_coding_agreement import (
    nominal_krippendorff_alpha,
)
from scripts.analyze_qualitative_codes import (
    apply_adjudication,
    derive_variables,
)


ROOT = Path(__file__).resolve().parents[1]


def coding_rows():
    return pd.DataFrame(
        [
            {
                "coding_id": "c1",
                "geography_present": "yes",
                "evidence_status": (
                    "unsupported_general_inference"
                ),
                "direction": "negative",
                "decision_relevance": "yes",
                "coder_confidence": "high",
            },
            {
                "coding_id": "c2",
                "geography_present": "yes",
                "evidence_status": (
                    "neutral_restatement"
                ),
                "direction": "neutral",
                "decision_relevance": "no",
                "coder_confidence": "high",
            },
            {
                "coding_id": "c3",
                "geography_present": "no",
                "evidence_status": "none",
                "direction": "none",
                "decision_relevance": "no",
                "coder_confidence": "high",
            },
            {
                "coding_id": "c4",
                "geography_present": "yes",
                "evidence_status": (
                    "unsupplied_specific_geographic_claim"
                ),
                "direction": "positive",
                "decision_relevance": "yes",
                "coder_confidence": "medium",
            },
        ]
    )


def private_key_rows():
    return pd.DataFrame(
        [
            {
                "coding_id": "c1",
                "run_id": "r1",
                "model_id": "m1",
                "scenario_id": "s1",
                "strength": "strong",
                "condition": "labelled",
                "country": "Kyrgyzstan",
                "repetition": "1",
            },
            {
                "coding_id": "c2",
                "run_id": "r2",
                "model_id": "m1",
                "scenario_id": "s2",
                "strength": "borderline",
                "condition": "labelled",
                "country": "United States",
                "repetition": "1",
            },
            {
                "coding_id": "c3",
                "run_id": "r3",
                "model_id": "m2",
                "scenario_id": "s3",
                "strength": "weak",
                "condition": "anonymous",
                "country": "Anonymous",
                "repetition": "1",
            },
            {
                "coding_id": "c4",
                "run_id": "r4",
                "model_id": "m3",
                "scenario_id": "s4",
                "strength": "strong",
                "condition": "mitigation",
                "country": "Kyrgyzstan",
                "repetition": "1",
            },
        ]
    )


def run_script(*args):
    return subprocess.run(
        [
            sys.executable,
            *args,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_nominal_krippendorff_alpha_perfect_agreement():
    alpha = nominal_krippendorff_alpha(
        ["yes", "no", "yes", "no"],
        ["yes", "no", "yes", "no"],
    )

    assert alpha == 1.0


def test_nominal_krippendorff_alpha_undefined_without_variation():
    alpha = nominal_krippendorff_alpha(
        ["yes", "yes", "yes"],
        ["yes", "yes", "yes"],
    )

    assert math.isnan(alpha)


def test_derived_qualitative_variables_follow_codebook():
    frame = coding_rows()

    derived = derive_variables(
        frame
    )

    assert derived[
        "unsupported_geography"
    ].tolist() == [
        1,
        0,
        0,
        1,
    ]

    assert derived[
        "unsupported_negative"
    ].tolist() == [
        1,
        0,
        0,
        0,
    ]

    assert derived[
        "unsupported_positive"
    ].tolist() == [
        0,
        0,
        0,
        1,
    ]

    assert derived[
        "geography_decision_relevant"
    ].tolist() == [
        1,
        0,
        0,
        1,
    ]


def test_completed_adjudication_overrides_researcher_code(
    tmp_path,
):
    researcher = coding_rows()

    adjudication = pd.DataFrame(
        [
            {
                "coding_id": "c4",
                "adjudicated_geography_present": "yes",
                "adjudicated_evidence_status": (
                    "unsupported_general_inference"
                ),
                "adjudicated_direction": "negative",
                "adjudicated_decision_relevance": "yes",
                "adjudicated_coder_confidence": "high",
            }
        ]
    )

    path = (
        tmp_path
        / "adjudication.csv"
    )

    adjudication.to_csv(
        path,
        index=False,
    )

    final = apply_adjudication(
        researcher,
        path,
    )

    row = final.loc[
        final["coding_id"].eq("c4")
    ].iloc[0]

    assert (
        row["evidence_status"]
        == "unsupported_general_inference"
    )
    assert row["direction"] == "negative"
    assert row["coder_confidence"] == "high"


def test_postcoding_cli_pipeline_and_original_files_preserved(
    tmp_path,
):
    researcher = coding_rows()

    second = researcher[
        researcher[
            "coding_id"
        ].isin(
            [
                "c1",
                "c2",
                "c4",
            ]
        )
    ].copy()

    # Deliberately create exactly one disagreement.
    second.loc[
        second[
            "coding_id"
        ].eq("c4"),
        "evidence_status",
    ] = (
        "unsupported_general_inference"
    )

    private = private_key_rows()

    researcher_path = (
        tmp_path
        / "researcher.csv"
    )
    second_path = (
        tmp_path
        / "second.csv"
    )
    private_path = (
        tmp_path
        / "private.csv"
    )

    researcher.to_csv(
        researcher_path,
        index=False,
    )
    second.to_csv(
        second_path,
        index=False,
    )
    private.to_csv(
        private_path,
        index=False,
    )

    researcher_before = (
        researcher_path.read_bytes()
    )
    second_before = (
        second_path.read_bytes()
    )

    agreement_dir = (
        tmp_path
        / "agreement"
    )

    agreement_result = run_script(
        "scripts/analyze_coding_agreement.py",
        "--researcher",
        str(researcher_path),
        "--second-coder",
        str(second_path),
        "--out-dir",
        str(agreement_dir),
    )

    assert (
        "Double-coded items: 3"
        in agreement_result.stdout
    )

    agreement = pd.read_csv(
        agreement_dir
        / "agreement_summary.csv"
    )

    evidence = agreement.loc[
        agreement[
            "dimension"
        ].eq(
            "evidence_status"
        )
    ].iloc[0]

    assert evidence[
        "agreement_count"
    ] == 2

    assert evidence[
        "disagreement_count"
    ] == 1

    adjudication_path = (
        tmp_path
        / "adjudication.csv"
    )

    adjudication_result = run_script(
        "scripts/prepare_adjudication_sheet.py",
        "--researcher",
        str(researcher_path),
        "--second-coder",
        str(second_path),
        "--output",
        str(adjudication_path),
    )

    assert (
        "Items with >=1 disagreement: 1"
        in adjudication_result.stdout
    )

    adjudication = pd.read_csv(
        adjudication_path
    )

    assert len(adjudication) == 1

    assert (
        adjudication.iloc[0][
            "coding_id"
        ]
        == "c4"
    )

    qualitative_dir = (
        tmp_path
        / "qualitative"
    )

    qualitative_result = run_script(
        "scripts/analyze_qualitative_codes.py",
        "--researcher",
        str(researcher_path),
        "--private-key",
        str(private_path),
        "--adjudication",
        str(
            tmp_path
            / "no_adjudication.csv"
        ),
        "--out-dir",
        str(qualitative_dir),
    )

    assert (
        "Eligible coded items: 4"
        in qualitative_result.stdout
    )

    rates = pd.read_csv(
        qualitative_dir
        / "qualitative_rates_by_condition.csv"
    )

    comparison = pd.read_csv(
        qualitative_dir
        / (
            "unsupported_geography_"
            "labelled_vs_mitigation.csv"
        )
    )

    assert len(rates) == 12

    labelled = comparison.loc[
        comparison[
            "condition"
        ].eq(
            "labelled"
        )
    ].iloc[0]

    mitigation = comparison.loc[
        comparison[
            "condition"
        ].eq(
            "mitigation"
        )
    ].iloc[0]

    assert (
        labelled[
            "unsupported_numerator"
        ]
        == 1
    )

    assert (
        labelled["denominator"]
        == 2
    )

    assert (
        mitigation[
            "unsupported_numerator"
        ]
        == 1
    )

    assert (
        mitigation["denominator"]
        == 1
    )

    assert (
        mitigation[
            "mitigation_minus_labelled_pp"
        ]
        == 50.0
    )

    # Critical audit rule:
    # neither original coder file may be overwritten.
    assert (
        researcher_path.read_bytes()
        == researcher_before
    )

    assert (
        second_path.read_bytes()
        == second_before
    )
