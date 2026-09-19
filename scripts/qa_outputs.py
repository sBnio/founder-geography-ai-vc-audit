from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from common import ROOT, load_json


EXPECTED_FIELDS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check model-output quality without calling any APIs."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/pilot/pilot_responses.jsonl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/pilot/pilot_qa_report.csv",
    )

    parser.add_argument(
        "--current-models-only",
        action="store_true",
        help="Keep only models currently configured in experiment.json.",
    )

    parser.add_argument(
        "--expect-runs",
        type=int,
        default=None,
        help="Fail if the number of unique latest runs differs.",
    )

    return parser.parse_args()


def load_evaluation_model():
    possible_modules = [
        "schemas",
        "scripts.schemas",
    ]

    possible_names = [
        "Evaluation",
        "EvaluationResponse",
        "ModelEvaluation",
    ]

    last_error: Exception | None = None

    for module_name in possible_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            last_error = exc
            continue

        for class_name in possible_names:
            model = getattr(module, class_name, None)

            if model is not None:
                return model

    raise RuntimeError(
        "Could not find the local Evaluation Pydantic model "
        f"in scripts/schemas.py. Last import error: {last_error}"
    )


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    return cleaned


def looks_like_evaluation(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "decision" in value
        and "investment_score_0_100" in value
    )


def extract_payload(
    value: Any,
    depth: int = 0,
) -> dict[str, Any] | None:
    if depth > 5:
        return None

    if looks_like_evaluation(value):
        return value

    if isinstance(value, str):
        cleaned = strip_code_fence(value)

        try:
            decoded = json.loads(cleaned)
        except Exception:
            return None

        return extract_payload(decoded, depth + 1)

    if isinstance(value, dict):
        priority_keys = [
            "parsed_response",
            "response_json",
            "response",
            "parsed",
            "output",
            "result",
            "content",
            "text",
            "raw_response_text",
            "response_text",
        ]

        for key in priority_keys:
            if key not in value:
                continue

            found = extract_payload(
                value[key],
                depth + 1,
            )

            if found is not None:
                return found

        for nested_value in value.values():
            found = extract_payload(
                nested_value,
                depth + 1,
            )

            if found is not None:
                return found

    if isinstance(value, list):
        for item in value:
            found = extract_payload(
                item,
                depth + 1,
            )

            if found is not None:
                return found

    return None


def extract_actual_model(record: dict[str, Any]) -> str | None:
    direct_candidates = [
        record.get("actual_model_returned"),
        record.get("returned_model"),
    ]

    for candidate in direct_candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    metadata = record.get("provider_metadata")

    if not isinstance(metadata, dict):
        return None

    keys = [
        "actual_model_returned",
        "returned_model",
        "model_id",
        "model",
    ]

    for key in keys:
        candidate = metadata.get(key)

        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    for value in metadata.values():
        if not isinstance(value, dict):
            continue

        for key in keys:
            candidate = value.get(key)

            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    return None


def validate_with_pydantic(
    evaluation_model,
    payload: dict[str, Any],
) -> tuple[bool, str]:
    try:
        if hasattr(evaluation_model, "model_validate"):
            evaluation_model.model_validate(payload)
        elif hasattr(evaluation_model, "parse_obj"):
            evaluation_model.parse_obj(payload)
        else:
            raise RuntimeError(
                "Evaluation model has no supported validation method."
            )

        return True, ""

    except Exception as exc:
        message = str(exc).replace("\n", " ")

        if len(message) > 800:
            message = message[:800] + "..."

        return False, message


def is_number_in_range(
    value: Any,
    lower: float,
    upper: float,
) -> bool:
    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return lower <= float(value) <= upper


def word_count(value: Any) -> int:
    if not isinstance(value, str):
        return 0

    return len(
        [
            word
            for word in value.strip().split()
            if word
        ]
    )


def canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def latest_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(records):
        run_id = record.get("run_id")

        if not isinstance(run_id, str) or not run_id:
            run_id = f"MISSING_RUN_ID_{index}"

        attempt = record.get("attempt", 0)

        try:
            attempt_number = int(attempt)
        except Exception:
            attempt_number = 0

        previous = latest.get(run_id)

        if previous is None:
            stored = dict(record)
            stored["_qa_attempt_number"] = attempt_number
            latest[run_id] = stored
            continue

        previous_attempt = previous.get(
            "_qa_attempt_number",
            0,
        )

        if attempt_number >= previous_attempt:
            stored = dict(record)
            stored["_qa_attempt_number"] = attempt_number
            latest[run_id] = stored

    return list(latest.values())


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    config = load_json(
        ROOT / "config/experiment.json"
    )

    final_cases = load_json(
        ROOT / "scenarios/final_cases.json"
    )

    requested_raise_by_scenario = {
        case["id"]: int(case["raise_usd"])
        for case in final_cases
    }

    selected_models = {
        (
            model.get("provider"),
            model.get("model_id"),
        )
        for model in config.get("models", [])
    }

    raw_records: list[dict[str, Any]] = []

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except Exception as exc:
                print(
                    f"WARNING: Invalid JSONL line "
                    f"{line_number}: {exc}"
                )
                continue

            if not isinstance(record, dict):
                continue

            if args.current_models_only:
                key = (
                    record.get("provider"),
                    record.get("model_id"),
                )

                if key not in selected_models:
                    continue

            raw_records.append(record)

    records = latest_records(raw_records)

    if args.expect_runs is not None:
        if len(records) != args.expect_runs:
            print(
                "ERROR: Unexpected unique-run count. "
                f"Expected {args.expect_runs}, "
                f"found {len(records)}."
            )
            return 1

    evaluation_model = load_evaluation_model()

    report_rows: list[dict[str, Any]] = []
    response_hashes: Counter[str] = Counter()

    for record in records:
        status = str(
            record.get("status", "")
        ).strip().lower()

        payload = None

        if status == "success":
            payload = extract_payload(record)

        payload_found = payload is not None

        if payload is not None:
            response_hash = canonical_hash(payload)
            response_hashes[response_hash] += 1
        else:
            response_hash = ""

        schema_valid = False
        validation_error = ""

        if payload is not None:
            schema_valid, validation_error = (
                validate_with_pydantic(
                    evaluation_model,
                    payload,
                )
            )

        factors = (
            payload.get("decision_factors", [])
            if payload
            else []
        )

        assumptions = (
            payload.get(
                "assumptions_or_inferences",
                [],
            )
            if payload
            else []
        )

        missing_information = (
            payload.get(
                "missing_information",
                [],
            )
            if payload
            else []
        )

        rationale = (
            payload.get(
                "one_sentence_rationale",
                "",
            )
            if payload
            else ""
        )

        factor_count_valid = (
            isinstance(factors, list)
            and 3 <= len(factors) <= 4
        )

        assumptions_valid = (
            isinstance(assumptions, list)
            and len(assumptions) <= 2
        )

        missing_information_valid = (
            isinstance(missing_information, list)
            and len(missing_information) <= 3
        )

        rationale_words = word_count(rationale)

        rationale_valid = (
            isinstance(rationale, str)
            and 1 <= rationale_words <= 35
        )

        factor_word_limits_valid = True
        factor_word_limit_warning = False

        if not isinstance(factors, list):
            factor_word_limits_valid = False
        else:
            for factor in factors:
                if not isinstance(factor, dict):
                    factor_word_limits_valid = False
                    break

                factor_text = factor.get("factor", "")

                if not (
                    isinstance(factor_text, str)
                    and 1 <= word_count(factor_text) <= 8
                ):
                    factor_word_limits_valid = False

                if (
                    isinstance(factor_text, str)
                    and word_count(factor_text) > 8
                ):
                    factor_word_limit_warning = True

        rationale_word_limit_warning = (
            isinstance(rationale, str)
            and rationale_words > 35
        )

        numeric_ranges_valid = False

        if payload is not None:
            numeric_ranges_valid = all(
                [
                    is_number_in_range(
                        payload.get(
                            "investment_score_0_100"
                        ),
                        0,
                        100,
                    ),
                    is_number_in_range(
                        payload.get(
                            "probability_of_closing_stated_round_within_12_months_percent"
                        ),
                        0,
                        100,
                    ),
                    is_number_in_range(
                        payload.get(
                            "financial_risk_score_0_100"
                        ),
                        0,
                        100,
                    ),
                    is_number_in_range(
                        payload.get(
                            "confidence_0_100"
                        ),
                        0,
                        100,
                    ),
                ]
            )

        extra_fields: list[str] = []

        if payload is not None:
            extra_fields = sorted(
                set(payload.keys())
                - EXPECTED_FIELDS
            )

        no_extra_fields = len(extra_fields) == 0

        configured_model = str(
            record.get("model_id", "")
        )

        actual_model = extract_actual_model(
            record
        )

        if actual_model is None:
            model_match = "NOT_REPORTED"
            model_match_valid = True
        elif actual_model == configured_model:
            model_match = "YES"
            model_match_valid = True
        else:
            model_match = "NO"
            model_match_valid = False

        requested_raise = requested_raise_by_scenario.get(
            record.get("scenario_id")
        )

        decision = (
            payload.get("decision")
            if payload
            else None
        )

        recommended_investment = (
            payload.get("recommended_investment_usd")
            if payload
            else None
        )

        pass_nonzero_warning = bool(
            payload
            and decision == "pass"
            and isinstance(recommended_investment, int)
            and not isinstance(recommended_investment, bool)
            and recommended_investment != 0
        )

        investment_above_raise_warning = bool(
            payload
            and isinstance(recommended_investment, int)
            and not isinstance(recommended_investment, bool)
            and isinstance(requested_raise, int)
            and recommended_investment > requested_raise
        )

        row = {
            "run_id": record.get("run_id", ""),
            "attempt": record.get("attempt", ""),
            "provider": record.get("provider", ""),
            "model_id": configured_model,
            "actual_model_returned": (
                actual_model or ""
            ),
            "scenario_id": record.get(
                "scenario_id",
                "",
            ),
            "condition": record.get(
                "condition",
                "",
            ),
            "country": record.get("country", ""),
            "status": status,
            "payload_found": payload_found,
            "schema_valid": schema_valid,
            "factor_count_valid": factor_count_valid,
            "factor_word_limits_valid": (
                factor_word_limits_valid
            ),
            "assumptions_valid": assumptions_valid,
            "missing_information_valid": (
                missing_information_valid
            ),
            "rationale_word_count": rationale_words,
            "rationale_valid": rationale_valid,
            "factor_word_limit_warning": factor_word_limit_warning,
            "rationale_word_limit_warning": rationale_word_limit_warning,
            "requested_raise_usd": (
                requested_raise
                if requested_raise is not None
                else ""
            ),
            "pass_nonzero_warning": pass_nonzero_warning,
            "investment_above_raise_warning": (
                investment_above_raise_warning
            ),
            "numeric_ranges_valid": (
                numeric_ranges_valid
            ),
            "no_extra_fields": no_extra_fields,
            "extra_fields": "; ".join(
                extra_fields
            ),
            "model_match": model_match,
            "response_hash": response_hash,
            "duplicate_response": False,
            "schema_validation_error": (
                validation_error
            ),
            "error_type": record.get(
                "error_type",
                "",
            ),
            "error": record.get("error", ""),
            "qa_warning": False,
            "manual_review_required": False,
            "qa_pass": False,
            "qa_notes": "",
        }

        report_rows.append(row)

    for row in report_rows:
        response_hash = row["response_hash"]

        duplicate = bool(
            response_hash
            and response_hashes[response_hash] > 1
        )

        row["duplicate_response"] = duplicate

        required_checks = [
            row["status"] == "success",
            row["payload_found"],
            row["schema_valid"],
            row["factor_count_valid"],
            row["assumptions_valid"],
            row["missing_information_valid"],
            row["numeric_ranges_valid"],
            row["no_extra_fields"],
            row["model_match"] != "NO",
        ]

        qa_pass = all(required_checks)

        notes: list[str] = []

        if row["status"] != "success":
            notes.append("Latest attempt is not successful")

        if not row["payload_found"]:
            notes.append(
                "Structured evaluation payload not found"
            )

        if (
            row["payload_found"]
            and not row["schema_valid"]
        ):
            notes.append(
                "Failed local Pydantic validation"
            )

        if not row["factor_count_valid"]:
            notes.append(
                "Decision factors must total 3 or 4"
            )

        if not row["assumptions_valid"]:
            notes.append(
                "More than 2 assumptions "
                "or malformed list"
            )

        if not row["missing_information_valid"]:
            notes.append(
                "More than 3 missing-information "
                "items or malformed list"
            )

        if not row["numeric_ranges_valid"]:
            notes.append(
                "One or more 0–100 fields "
                "are invalid"
            )

        if not row["no_extra_fields"]:
            notes.append(
                "Unexpected top-level fields present"
            )

        if row["model_match"] == "NO":
            notes.append(
                "Returned model differs from "
                "configured model"
            )

        if row["factor_word_limit_warning"]:
            notes.append(
                "WARNING retained: factor wording exceeds 8 words"
            )

        if row["rationale_word_limit_warning"]:
            notes.append(
                "WARNING retained: rationale exceeds 35 words"
            )

        if row["pass_nonzero_warning"]:
            notes.append(
                "WARNING retained: pass decision has nonzero investment"
            )

        if row["investment_above_raise_warning"]:
            notes.append(
                "WARNING retained: recommended investment "
                "exceeds requested raise"
            )

        if duplicate:
            notes.append(
                "WARNING retained: exact duplicate response"
            )

        warning_present = any(
            [
                row["factor_word_limit_warning"],
                row["rationale_word_limit_warning"],
                row["pass_nonzero_warning"],
                row["investment_above_raise_warning"],
                duplicate,
            ]
        )

        row["qa_warning"] = warning_present
        row["qa_pass"] = qa_pass

        row["manual_review_required"] = (
            not qa_pass
            or warning_present
        )

        row["qa_notes"] = "; ".join(notes)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "run_id",
        "attempt",
        "provider",
        "model_id",
        "actual_model_returned",
        "scenario_id",
        "condition",
        "country",
        "status",
        "payload_found",
        "schema_valid",
        "factor_count_valid",
        "factor_word_limits_valid",
        "assumptions_valid",
        "missing_information_valid",
        "rationale_word_count",
        "rationale_valid",
        "factor_word_limit_warning",
        "rationale_word_limit_warning",
        "requested_raise_usd",
        "pass_nonzero_warning",
        "investment_above_raise_warning",
        "numeric_ranges_valid",
        "no_extra_fields",
        "extra_fields",
        "model_match",
        "duplicate_response",
        "qa_warning",
        "manual_review_required",
        "qa_pass",
        "qa_notes",
        "schema_validation_error",
        "error_type",
        "error",
        "response_hash",
    ]

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(report_rows)

    status_counts = Counter(
        row["status"]
        for row in report_rows
    )

    schema_valid_count = sum(
        bool(row["schema_valid"])
        for row in report_rows
    )

    qa_pass_count = sum(
        bool(row["qa_pass"])
        for row in report_rows
    )

    duplicate_count = sum(
        bool(row["duplicate_response"])
        for row in report_rows
    )

    warning_count = sum(
        bool(row["qa_warning"])
        for row in report_rows
    )

    review_count = sum(
        bool(row["manual_review_required"])
        for row in report_rows
    )

    print("=" * 76)
    print("OUTPUT QUALITY-CONTROL REPORT")
    print("=" * 76)

    print(
        f"Raw records considered:       "
        f"{len(raw_records)}"
    )

    print(
        f"Latest unique runs checked:   "
        f"{len(report_rows)}"
    )

    print(
        f"Successful latest runs:       "
        f"{status_counts.get('success', 0)}"
    )

    print(
        f"Locally schema-valid:         "
        f"{schema_valid_count}"
    )

    print(
        f"Full QA passes:               "
        f"{qa_pass_count}"
    )

    print(
        f"Exact duplicate responses:    "
        f"{duplicate_count}"
    )

    print(
        f"Retained QA warnings:         "
        f"{warning_count}"
    )

    print(
        f"Manual-review flags:          "
        f"{review_count}"
    )

    print(
        f"Report saved to:              "
        f"{args.output}"
    )

    expected = len(report_rows)

    hard_pass = (
        expected > 0
        and status_counts.get("success", 0)
        == expected
        and schema_valid_count == expected
        and qa_pass_count == expected
    )

    print()

    if hard_pass:
        print("RESULT: PASS")
        print(
            "Every latest run passed the "
            "automatic hard checks."
        )

        if warning_count:
            print(
                "Warnings were retained as model behaviour; "
                "they are not exclusion triggers."
            )

        return 0

    print("RESULT: FAIL")
    print(
        "Open the CSV and inspect rows where "
        "qa_pass is False."
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())
