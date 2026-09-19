import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.process import flatten, select_final_records


def test_select_final_records_prefers_success_and_latest_attempt():
    records = [
        {"run_id": "a", "attempt": 1, "status": "error", "timestamp_utc": "2026-01-01"},
        {"run_id": "a", "attempt": 2, "status": "success", "timestamp_utc": "2026-01-02", "response": {}},
        {"run_id": "a", "attempt": 3, "status": "error", "timestamp_utc": "2026-01-03"},
        {"run_id": "b", "attempt": 1, "status": "error", "timestamp_utc": "2026-01-01"},
        {"run_id": "b", "attempt": 2, "status": "error", "timestamp_utc": "2026-01-02"},
    ]
    selected = {record["run_id"]: record for record in select_final_records(records)}
    assert selected["a"]["status"] == "success"
    assert selected["a"]["attempt"] == 2
    assert selected["b"]["attempt"] == 2


def test_flatten_serializes_nested_fields_and_usage():
    record = {
        "run_id": "x",
        "status": "success",
        "response": {
            "decision": "pass",
            "decision_factors": [{"factor": "x"}],
            "assumptions_or_inferences": ["a"],
            "missing_information": ["m"],
        },
        "provider_metadata": {
            "actual_model_returned": "model-x",
            "request_id": "req-1",
            "standard_usage": {
                "input_tokens": 100,
                "visible_output_tokens": 50,
                "reasoning_tokens": 25,
                "total_tokens": 175,
                "estimated_cost_usd": 0.01,
            },
        },
    }
    row = flatten(record)
    assert json.loads(row["decision_factors_json"])[0]["factor"] == "x"
    assert "decision_factors" not in row
    assert row["input_tokens"] == 100
    assert row["actual_model_returned"] == "model-x"


def test_retry_policy_only_retries_technical_failures():
    from scripts.collect import is_retryable_exception
    from providers.adapters import ProviderCallError

    json_error = json.JSONDecodeError(
        "bad json",
        "",
        0,
    )

    assert is_retryable_exception(json_error)

    assert is_retryable_exception(
        ProviderCallError("HTTP 503: temporary outage")
    )

    assert not is_retryable_exception(
        ProviderCallError(
            "Missing required environment variable: TEST_KEY"
        )
    )

    assert not is_retryable_exception(
        RuntimeError("configuration mismatch")
    )
