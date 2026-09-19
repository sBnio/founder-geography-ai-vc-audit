import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from common import ROOT
except ModuleNotFoundError:
    from scripts.common import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert append-only JSONL responses into a deterministic tidy CSV.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/responses.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/responses.csv")
    return parser.parse_args()


def select_final_records(records: list[dict]) -> list[dict]:
    by_run: dict[str, list[dict]] = {}
    for record in records:
        run_id = record.get("run_id")
        if not run_id:
            continue
        by_run.setdefault(run_id, []).append(record)
    selected: list[dict] = []
    for run_records in by_run.values():
        run_records.sort(key=lambda r: (int(r.get("attempt", 1)), r.get("timestamp_utc", "")))
        successes = [record for record in run_records if record.get("status") == "success"]
        selected.append(successes[-1] if successes else run_records[-1])
    return selected


def flatten(record: dict) -> dict:
    base_keys = [
        "run_id",
        "condition",
        "scenario_id",
        "strength",
        "country",
        "provider",
        "model_id",
        "repetition",
        "timestamp_utc",
        "attempt",
        "status",
        "latency_seconds",
        "prompt_hash",
        "error_type",
        "error",
        "raw_response_text",
    ]
    row = {key: record.get(key) for key in base_keys}
    response = record.get("response") or {}
    row.update(response)
    row["decision_factors_json"] = json.dumps(response.get("decision_factors", []), ensure_ascii=False)
    row["assumptions_or_inferences_json"] = json.dumps(response.get("assumptions_or_inferences", []), ensure_ascii=False)
    row["missing_information_json"] = json.dumps(response.get("missing_information", []), ensure_ascii=False)
    row.pop("decision_factors", None)
    row.pop("assumptions_or_inferences", None)
    row.pop("missing_information", None)

    metadata = record.get("provider_metadata") or {}
    usage = metadata.get("standard_usage") or {}
    row.update(
        {
            "actual_model_returned": metadata.get("actual_model_returned"),
            "provider_request_id": metadata.get("request_id"),
            "input_tokens": usage.get("input_tokens"),
            "visible_output_tokens": usage.get("visible_output_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "estimated_cost_usd": usage.get("estimated_cost_usd"),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input response log not found: {args.input}")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_final_records(records)
    dataframe = pd.DataFrame([flatten(record) for record in selected])
    if dataframe.empty:
        raise RuntimeError("No valid records found")
    dataframe = dataframe.sort_values(
        ["condition", "scenario_id", "model_id", "country", "repetition"], na_position="last"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output, index=False)
    success = int((dataframe["status"] == "success").sum())
    cost = pd.to_numeric(dataframe.get("estimated_cost_usd"), errors="coerce").sum(min_count=1)
    print(
        {
            "raw_records": len(records),
            "unique_runs": len(dataframe),
            "success": success,
            "errors": len(dataframe) - success,
            "estimated_cost_usd": None if pd.isna(cost) else round(float(cost), 6),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
