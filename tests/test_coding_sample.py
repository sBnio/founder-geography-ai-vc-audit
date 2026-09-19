import pandas as pd

from scripts.build_blinded_coding_sample import (
    LOCATION_TOKEN,
    STRATIFICATION_COLUMNS,
    allocate_stratified_counts,
    direct_treatment_leaks,
    make_coder_sheet,
    mask_location_terms,
    target_second_coder_sample_size,
)


def test_second_coder_sample_size_rule():
    assert target_second_coder_sample_size(360) == 72
    assert target_second_coder_sample_size(359) == 72
    assert target_second_coder_sample_size(300) == 60
    assert target_second_coder_sample_size(100) == 20


def test_stratified_allocation_is_exact_and_covers_strata_when_possible():
    strata = pd.DataFrame(
        [
            {
                "condition": "labelled",
                "model_id": "M1",
                "country": "Kyrgyzstan",
                "strength": "Strong",
                "eligible_n": 10,
            },
            {
                "condition": "labelled",
                "model_id": "M1",
                "country": "United States",
                "strength": "Strong",
                "eligible_n": 20,
            },
            {
                "condition": "anonymous",
                "model_id": "M1",
                "country": "Anonymous",
                "strength": "Weak",
                "eligible_n": 30,
            },
        ]
    )

    result = allocate_stratified_counts(
        strata,
        12,
    )

    assert result["selected_n"].sum() == 12
    assert (result["selected_n"] >= 1).all()
    assert (
        result["selected_n"]
        <= result["eligible_n"]
    ).all()


def test_masking_and_coder_sheet_remove_direct_treatment_identity():
    assert (
        mask_location_terms(
            "Kyrgyz founders compared with the United States and USA."
        )
        .count(LOCATION_TOKEN)
        == 3
    )

    # Lowercase English pronoun must not be mistaken for "US".
    assert mask_location_terms(
        "This gives us more information."
    ) == "This gives us more information."

    rows = pd.DataFrame(
        [
            {
                "coding_id": "C0001",
                "scenario_id": "S1",
                "condition": "labelled",
                "decision_factors_json": (
                    '["Kyrgyzstan investor access"]'
                ),
                "assumptions_or_inferences_json": "[]",
                "missing_information_json": "[]",
                "one_sentence_rationale": (
                    "American fundraising may be easier."
                ),
            }
        ]
    )

    case_map = {
        "S1": {
            "pitch_template": (
                "Example startup. "
                "The founding team currently lives and "
                "works remotely from [COUNTRY]."
            )
        }
    }

    sheet = make_coder_sheet(
        rows,
        case_map,
    )

    assert not direct_treatment_leaks(sheet)
    assert "model_id" not in sheet.columns
    assert "country" not in sheet.columns
    assert "decision" not in sheet.columns
    assert "investment_score_0_100" not in sheet.columns

    assert set(
        [
            "geography_present",
            "evidence_status",
            "direction",
            "decision_relevance",
            "coder_confidence",
        ]
    ).issubset(sheet.columns)
