from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Factor(StrictModel):
    factor: str
    direction: Literal["positive", "negative"]
    importance_1_5: int = Field(ge=1, le=5)

    @field_validator("factor")
    @classmethod
    def factor_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("factor must not be empty")
        return value


class Evaluation(StrictModel):
    decision: Literal["invest", "pass", "request_more_information"]
    investment_score_0_100: int = Field(ge=0, le=100)
    recommended_investment_usd: int = Field(ge=0)
    recommended_pre_money_valuation_usd: int = Field(ge=0)
    probability_of_closing_stated_round_within_12_months_percent: int = Field(ge=0, le=100)
    financial_risk_score_0_100: int = Field(ge=0, le=100)
    confidence_0_100: int = Field(ge=0, le=100)
    decision_factors: list[Factor] = Field(min_length=3, max_length=4)
    assumptions_or_inferences: list[str] = Field(max_length=2)
    missing_information: list[str] = Field(max_length=3)
    one_sentence_rationale: str

    @field_validator("assumptions_or_inferences", "missing_information")
    @classmethod
    def nonempty_list_items(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value.strip():
                raise ValueError("list items must not be empty")
        return values

    @field_validator("one_sentence_rationale")
    @classmethod
    def rationale_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("rationale must not be empty")
        return value


def evaluation_api_schema() -> dict:
    """Provider-compatible structural JSON Schema."""
    factor_schema = {
        "type": "object",
        "properties": {
            "factor": {"type": "string", "minLength": 1, "description": "Concise decision factor."},
            "direction": {"type": "string", "enum": ["positive", "negative"]},
            "importance_1_5": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Importance as an integer from 1 to 5.",
            },
        },
        "required": ["factor", "direction", "importance_1_5"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["invest", "pass", "request_more_information"],
            },
            "investment_score_0_100": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Investment score from 0 to 100.",
            },
            "recommended_investment_usd": {
                "type": "integer",
                "minimum": 0,
                "description": "Recommended investment in USD; zero or greater.",
            },
            "recommended_pre_money_valuation_usd": {
                "type": "integer",
                "minimum": 0,
                "description": "Recommended pre-money valuation in USD; zero or greater.",
            },
            "probability_of_closing_stated_round_within_12_months_percent": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Probability from 0 to 100 that the startup closes the stated funding round within 12 months.",
            },
            "financial_risk_score_0_100": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Financial risk score from 0 to 100.",
            },
            "confidence_0_100": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Confidence score from 0 to 100.",
            },
            "decision_factors": {
                "type": "array",
                "description": "Provide 3 or 4 factors.",
                "items": factor_schema,
                "minItems": 3,
                "maxItems": 4,
            },
            "assumptions_or_inferences": {
                "type": "array",
                "description": "At most 2 short assumptions or inferences.",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 2,
            },
            "missing_information": {
                "type": "array",
                "description": "At most 3 short missing-information items.",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 3,
            },
            "one_sentence_rationale": {
                "type": "string",
                "minLength": 1,
                "description": "One concise sentence.",
            },
        },
        "required": [
            "decision",
            "investment_score_0_100",
            "recommended_investment_usd",
            "recommended_pre_money_valuation_usd",
            "probability_of_closing_stated_round_within_12_months_percent",
            "financial_risk_score_0_100",
            "confidence_0_100",
            "decision_factors",
            "assumptions_or_inferences",
            "missing_information",
            "one_sentence_rationale",
        ],
        "additionalProperties": False,
    }
