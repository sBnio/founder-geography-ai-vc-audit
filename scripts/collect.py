import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from pydantic import ValidationError

from scripts.common import ROOT, load_json
from providers.adapters import (
    ProviderCallError,
    call_anthropic,
    call_google,
    call_openai,
    call_openai_compatible,
)
from scripts.schemas import Evaluation

load_dotenv(ROOT / ".env")


def synthetic(row: dict) -> dict:
    """Create fake responses for software testing only."""
    model_offset = sum(ord(c) for c in row["model_id"]) % 7 - 3
    country_offset = {
        "Kyrgyzstan": -2,
        "United States": 2,
        "Anonymous": 0,
    }.get(row["country"], 0)

    base = {
        "Strong": 80,
        "Borderline": 55,
        "Weak": 28,
    }[row["strength"]]

    jitter = random.Random(row["run_id"]).randint(-4, 4)
    score = max(
        0,
        min(100, base + model_offset + country_offset + jitter),
    )

    return {
        "decision": (
            "invest"
            if score >= 70
            else "request_more_information"
            if score >= 45
            else "pass"
        ),
        "investment_score_0_100": score,
        "recommended_investment_usd": 500000 if score >= 70 else 0,
        "recommended_pre_money_valuation_usd": (
            5000000 if score >= 70 else 0
        ),
        "probability_of_closing_stated_round_within_12_months_percent": max(
            0,
            min(100, score - 10),
        ),
        "financial_risk_score_0_100": 100 - score,
        "confidence_0_100": 70,
        "decision_factors": [
            {
                "factor": "Synthetic output validates software only",
                "direction": "positive" if score >= 50 else "negative",
                "importance_1_5": 3,
            },
            {
                "factor": "Synthetic score follows case strength",
                "direction": "positive" if score >= 50 else "negative",
                "importance_1_5": 3,
            },
            {
                "factor": "Not usable as research evidence",
                "direction": "negative",
                "importance_1_5": 5,
            },
        ],
        "assumptions_or_inferences": [],
        "missing_information": [
            "Real model evaluation is intentionally absent"
        ],
        "one_sentence_rationale": (
            "This synthetic response tests collection, validation, "
            "processing, and analysis only."
        ),
    }


def parse_json(text: str) -> dict:
    """Parse JSON and remove accidental Markdown code fences."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    return json.loads(text)


def existing_state(path: Path) -> tuple[set[str], dict[str, int]]:
    """Read previous attempts so collection can resume safely."""
    successful: set[str] = set()
    attempts: dict[str, int] = {}

    if not path.exists():
        return successful, attempts

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        run_id = record.get("run_id")
        if not run_id:
            continue

        attempts[run_id] = max(
            attempts.get(run_id, 0),
            int(record.get("attempt", 1)),
        )

        if record.get("status") == "success":
            successful.add(run_id)

    return successful, attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect model responses while preserving an append-only log."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/run_manifest.jsonl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/raw/responses.jsonl",
    )

    parser.add_argument(
        "--mode",
        choices=["config", "synthetic", "pilot"],
        default="config",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional excluded-pilot limit.",
    )

    parser.add_argument(
        "--confirm-live",
        type=str,
        default=None,
        help=(
            "Exact explicit confirmation phrase required "
            "only for frozen final collection."
        ),
    )

    return parser.parse_args()



def is_retryable_exception(exc: Exception) -> bool:
    """Return True only for frozen technical retry categories."""

    if isinstance(exc, ProviderCallError):
        # Missing credentials/configuration must be fixed, not sampled again.
        if str(exc).startswith(
            "Missing required environment variable:"
        ):
            return False
        return True

    if isinstance(exc, json.JSONDecodeError):
        return True

    if isinstance(exc, ValidationError):
        return True

    return False


FINAL_LIVE_CONFIRMATION = "RUN-FROZEN-GEOVC-FINAL"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_manifest_sidecar(
    manifest_path: Path,
) -> None:
    if not manifest_path.exists():
        raise RuntimeError(
            f"Final manifest does not exist: {manifest_path}"
        )

    sidecar_path = Path(
        str(manifest_path) + ".sha256"
    )

    if not sidecar_path.exists():
        raise RuntimeError(
            "Final manifest SHA-256 sidecar is missing: "
            f"{sidecar_path}"
        )

    parts = sidecar_path.read_text(
        encoding="utf-8"
    ).strip().split()

    if not parts:
        raise RuntimeError(
            "Final manifest SHA-256 sidecar is empty."
        )

    expected = parts[0].strip().lower()
    actual = sha256_file(
        manifest_path
    ).lower()

    if expected != actual:
        raise RuntimeError(
            "Final manifest SHA-256 mismatch. "
            "Refusing live collection."
        )


def verify_osf_confirmation() -> None:
    path = (
        ROOT
        / "docs"
        / "OSF_REGISTRATION_CONFIRMED.txt"
    )

    if not path.exists():
        raise RuntimeError(
            "OSF registration confirmation is missing."
        )

    values = {}

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split(
            "=",
            1,
        )

        values[key.strip()] = value.strip()

    if (
        values.get(
            "CONFIRMED_BY_RESEARCHER"
        )
        != "YES"
    ):
        raise RuntimeError(
            "OSF registration has not been "
            "personally confirmed by the researcher."
        )

    osf_id = values.get(
        "OSF_URL_OR_ID",
        "",
    ).strip()

    confirmed_at = values.get(
        "CONFIRMED_AT_UTC",
        "",
    ).strip()

    if (
        not osf_id
        or "FILL_" in osf_id
        or "PLACEHOLDER" in osf_id.upper()
    ):
        raise RuntimeError(
            "OSF_URL_OR_ID is missing or still a placeholder."
        )

    if (
        not confirmed_at
        or "FILL_" in confirmed_at
        or "PLACEHOLDER" in confirmed_at.upper()
    ):
        raise RuntimeError(
            "CONFIRMED_AT_UTC is missing or "
            "still a placeholder."
        )


def verify_scientific_freeze() -> None:
    manifest = (
        ROOT
        / "docs"
        / "FREEZE_MANIFEST.json"
    )

    verifier = (
        ROOT
        / "scripts"
        / "verify_freeze.py"
    )

    if not manifest.exists():
        raise RuntimeError(
            "Scientific freeze manifest is missing."
        )

    if not verifier.exists():
        raise RuntimeError(
            "Scientific freeze verifier is missing."
        )

    result = subprocess.run(
        [
            sys.executable,
            str(verifier),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        details = (
            result.stdout
            + "\n"
            + result.stderr
        ).strip()

        raise RuntimeError(
            "Scientific freeze verification failed. "
            f"{details}"
        )


def require_final_live_gates(
    args: argparse.Namespace,
) -> None:
    """Fail closed before any final provider call."""

    if args.confirm_live != FINAL_LIVE_CONFIRMATION:
        raise RuntimeError(
            "Final collection requires exact confirmation: "
            f"{FINAL_LIVE_CONFIRMATION}"
        )

    if args.limit is not None:
        raise RuntimeError(
            "--limit is forbidden for final collection."
        )

    if args.max_attempts != 3:
        raise RuntimeError(
            "Final collection requires the frozen "
            "maximum of exactly 3 attempts."
        )

    runtime_path = (
        ROOT
        / "config"
        / "runtime.json"
    )

    if not runtime_path.exists():
        raise RuntimeError(
            "config/runtime.json is missing."
        )

    runtime = load_json(
        runtime_path
    )

    if (
        runtime.get(
            "live_collection_enabled"
        )
        is not True
    ):
        raise RuntimeError(
            "Final live collection is disabled in "
            "config/runtime.json."
        )

    verify_scientific_freeze()
    verify_osf_confirmation()
    verify_manifest_sidecar(
        args.manifest
    )


def main() -> None:
    args = parse_args()

    if args.mode == "config":
        require_final_live_gates(
            args
        )

    cfg = load_json(ROOT / "config/experiment.json")

    # Load final cases and separate P-series pilot cases.
    all_cases = load_json(ROOT / "scenarios/final_cases.json")

    pilot_cases_path = ROOT / "scenarios/pilot_cases.json"
    if pilot_cases_path.exists():
        all_cases.extend(load_json(pilot_cases_path))

    cases = {
        case["id"]: case
        for case in all_cases
    }

    model_cfg = {
        (model["provider"], model["model_id"]): model
        for model in cfg["models"]
    }

    successful, attempts = existing_state(args.output)

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if args.limit is not None:
        rows = rows[: args.limit]

    if args.mode == "config":
        if len(rows) != 360:
            raise RuntimeError(
                "Final manifest must contain exactly "
                f"360 runs; found {len(rows)}."
            )

        allowed_conditions = {
            "labelled",
            "anonymous",
            "mitigation",
        }

        if any(
            row.get("condition")
            not in allowed_conditions
            for row in rows
        ):
            raise RuntimeError(
                "Final manifest contains a "
                "non-final condition."
            )

        if any(
            str(
                row.get(
                    "scenario_id",
                    "",
                )
            ).upper().startswith("P")
            for row in rows
        ):
            raise RuntimeError(
                "Final manifest contains a pilot case."
            )

    adapters = {
        "openai": call_openai,
        "anthropic": call_anthropic,
        "google": call_google,
        "openai_compatible": call_openai_compatible,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_success = 0
    new_errors = 0

    # Validate manifest/configuration before the first provider call.
    for row in rows:
        if row["scenario_id"] not in cases:
            raise RuntimeError(
                "Fatal manifest error: unknown scenario ID "
                f"{row['scenario_id']}"
            )

        if (
            args.mode != "synthetic"
            and row["provider"] != "dry_run"
        ):
            key = (
                row["provider"],
                row["model_id"],
            )

            if key not in model_cfg:
                raise RuntimeError(
                    "Fatal model/configuration mismatch: "
                    f"{key}"
                )

            if row["provider"] not in adapters:
                raise RuntimeError(
                    "Fatal unsupported provider: "
                    f"{row['provider']}"
                )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "a",
        encoding="utf-8",
    ) as handle:

        # Pass-based retry schedule:
        # every eligible run receives attempt 1 before attempt 2 begins.
        for pass_number in range(
            1,
            args.max_attempts + 1,
        ):
            pending = [
                row
                for row in rows
                if (
                    row["run_id"] not in successful
                    and attempts.get(
                        row["run_id"],
                        0,
                    ) == pass_number - 1
                )
            ]

            if not pending:
                continue

            print(
                f"--- collection pass {pass_number}/"
                f"{args.max_attempts}: "
                f"{len(pending)} pending runs ---"
            )

            for index, row in enumerate(
                pending,
                1,
            ):
                run_id = row["run_id"]
                attempt = pass_number
                start_time = time.time()

                raw_response_text = None
                provider_metadata = None

                record = {
                    **row,
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "attempt": attempt,
                }

                fatal_exc = None

                try:
                    if (
                        args.mode == "synthetic"
                        or row["provider"] == "dry_run"
                    ):
                        obj = synthetic(row)

                        raw_response_text = json.dumps(
                            obj,
                            ensure_ascii=False,
                        )

                        provider_metadata = {
                            "synthetic": True,
                            "standard_usage": {
                                "input_tokens": None,
                                "visible_output_tokens": None,
                                "reasoning_tokens": None,
                                "total_tokens": None,
                                "estimated_cost_usd": 0.0,
                            },
                        }

                    else:
                        settings = model_cfg[
                            (
                                row["provider"],
                                row["model_id"],
                            )
                        ]

                        # Pilot mode may bypass enabled=false,
                        # but only for isolated P-series cases.
                        if args.mode == "pilot":
                            if row.get(
                                "condition"
                            ) != "pilot":
                                raise RuntimeError(
                                    "Pilot mode accepts only "
                                    "condition='pilot'"
                                )

                            if not str(
                                row.get(
                                    "scenario_id",
                                    "",
                                )
                            ).startswith("P"):
                                raise RuntimeError(
                                    "Pilot mode accepts only "
                                    "P-series pilot cases"
                                )

                        elif args.mode == "config":
                            # enabled=false remains part of the
                            # frozen scientific configuration.
                            # Operational permission was already
                            # established by require_final_live_gates().
                            pass

                        raw_response_text, provider_metadata = (
                            adapters[row["provider"]](
                                row,
                                settings,
                            )
                        )

                        obj = parse_json(
                            raw_response_text
                        )

                    valid = Evaluation.model_validate(
                        obj
                    )

                    record.update(
                        {
                            "status": "success",
                            "raw_response_text": (
                                raw_response_text
                            ),
                            "response": valid.model_dump(),
                            "provider_metadata": (
                                provider_metadata
                            ),
                            "error_type": "",
                            "error": "",
                        }
                    )

                    new_success += 1
                    successful.add(run_id)

                except Exception as exc:
                    if isinstance(
                        exc,
                        ProviderCallError,
                    ):
                        raw_response_text = (
                            exc.raw_response_text
                            or raw_response_text
                        )

                        provider_metadata = (
                            exc.provider_metadata
                            or provider_metadata
                        )

                    retryable = (
                        is_retryable_exception(exc)
                    )

                    record.update(
                        {
                            "status": "error",
                            "raw_response_text": (
                                raw_response_text
                            ),
                            "response": None,
                            "provider_metadata": (
                                provider_metadata
                            ),
                            "error_type": (
                                type(exc).__name__
                            ),
                            "error": str(exc),
                            "retryable": retryable,
                        }
                    )

                    new_errors += 1

                    if not retryable:
                        fatal_exc = exc

                record["latency_seconds"] = round(
                    time.time() - start_time,
                    3,
                )

                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                handle.flush()

                attempts[run_id] = attempt

                print(
                    f"pass={pass_number} "
                    f"{index}/{len(pending)} "
                    f"{record['status']} "
                    f"attempt={attempt} "
                    f"{row['provider']} "
                    f"{row['scenario_id']} "
                    f"{row['country']} "
                    f"{row['condition']}"
                )

                if fatal_exc is not None:
                    raise RuntimeError(
                        "FATAL collection error. "
                        "Collection stopped rather than "
                        "retrying a non-technical failure. "
                        f"run_id={run_id}"
                    ) from fatal_exc

    unresolved = [
        row["run_id"]
        for row in rows
        if row["run_id"] not in successful
    ]

    print(
        {
            "new_success": new_success,
            "new_errors": new_errors,
            "unresolved_after_max_attempts": (
                len(unresolved)
            ),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()