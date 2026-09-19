from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "analysis"),
)

from analyze import (
    build_completion_audit,
    summarize_completion,
)


def planned_fixture() -> pd.DataFrame:
    rows = []

    for index in range(3):
        rows.append(
            {
                "run_id": f"R{index + 1}",
                "condition": "labelled",
                "scenario_id": "S01",
                "strength": "Strong",
                "country": (
                    "Kyrgyzstan"
                    if index < 2
                    else "United States"
                ),
                "provider": "openai",
                "model_id": "test-model",
                "repetition": index + 1,
                "prompt_hash": f"hash-{index + 1}",
            }
        )

    return pd.DataFrame(
        rows
    )


def test_missing_planned_run_is_retained():
    planned = planned_fixture()

    processed = pd.DataFrame(
        [
            {
                "run_id": "R1",
                "status": "success",
                "attempt": 1,
                "latency_seconds": 1.0,
                "estimated_cost_usd": 0.1,
            },
            {
                "run_id": "R2",
                "status": "error",
                "attempt": 3,
                "latency_seconds": 2.0,
                "estimated_cost_usd": 0.0,
            },
        ]
    )

    audit = build_completion_audit(
        processed,
        planned,
    )

    assert len(audit) == 3

    status = dict(
        zip(
            audit["run_id"],
            audit["status"],
        )
    )

    assert status == {
        "R1": "success",
        "R2": "error",
        "R3": "missing",
    }

    summary = summarize_completion(
        audit,
        [
            "provider",
            "model_id",
        ],
    )

    row = summary.iloc[0]

    assert row["planned_runs"] == 3
    assert row["observed_runs"] == 2
    assert row["successful_runs"] == 1
    assert row["failed_runs"] == 1
    assert row["missing_runs"] == 1
    assert row["runs_requiring_retry"] == 1

    assert row["completion_rate"] == pytest.approx(
        2 / 3
    )

    assert row["missing_rate_of_planned"] == pytest.approx(
        1 / 3
    )


def test_unplanned_processed_run_is_fatal():
    planned = planned_fixture()

    processed = pd.DataFrame(
        [
            {
                "run_id": "NOT-IN-MANIFEST",
                "status": "success",
                "attempt": 1,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="absent from the planned manifest",
    ):
        build_completion_audit(
            processed,
            planned,
        )
