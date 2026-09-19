import pytest
from pydantic import ValidationError

from scripts.schemas import Evaluation, evaluation_api_schema


def valid_evaluation():
    return {
        "decision": "pass",
        "investment_score_0_100": 20,
        "recommended_investment_usd": 0,
        "recommended_pre_money_valuation_usd": 3000000,
        "probability_of_closing_stated_round_within_12_months_percent": 15,
        "financial_risk_score_0_100": 80,
        "confidence_0_100": 70,
        "decision_factors": [
            {"factor": "Weak retention", "direction": "negative", "importance_1_5": 5},
            {"factor": "Short runway", "direction": "negative", "importance_1_5": 4},
            {"factor": "Large market", "direction": "positive", "importance_1_5": 2},
        ],
        "assumptions_or_inferences": [],
        "missing_information": ["Cohort retention data"],
        "one_sentence_rationale": "Weak economics and limited runway make the investment unattractive despite possible market opportunity.",
    }


def test_compact_schema_accepts_valid_response():
    Evaluation.model_validate(valid_evaluation())


def test_compact_schema_rejects_extra_field():
    data = valid_evaluation()
    data["extra"] = "not allowed"
    with pytest.raises(ValidationError):
        Evaluation.model_validate(data)


def test_compact_schema_rejects_too_many_factors():
    data = valid_evaluation()
    data["decision_factors"] = data["decision_factors"] * 2
    with pytest.raises(ValidationError):
        Evaluation.model_validate(data)


def test_schema_accepts_semantically_inconsistent_but_structural_response():
    data = valid_evaluation()
    data["decision"] = "pass"
    data["recommended_investment_usd"] = 5000000

    validated = Evaluation.model_validate(data)

    assert validated.decision == "pass"
    assert validated.recommended_investment_usd == 5000000


def test_schema_accepts_stylistic_word_limit_violation():
    data = valid_evaluation()

    data["decision_factors"][0]["factor"] = (
        "This intentionally exceeds the requested eight word factor "
        "style limit but remains structurally valid"
    )

    data["one_sentence_rationale"] = " ".join(
        ["word"] * 40
    )

    validated = Evaluation.model_validate(data)

    assert validated.decision_factors[0].factor
    assert validated.one_sentence_rationale



def test_api_schema_does_not_request_basis():
    schema = evaluation_api_schema()
    factor_schema = schema["properties"]["decision_factors"]["items"]

    assert "basis" not in factor_schema["properties"]
    assert "basis" not in factor_schema["required"]
