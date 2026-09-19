from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter, defaultdict

from common import ROOT, load_json, sha256_text
from generate_manifest import generate_rows


EXPECTED_MODELS = {
    ("openai", "gpt-5.6-sol"),
    ("anthropic", "claude-opus-5"),
    ("google", "gemini-3.6-flash"),
}

EXPECTED_COUNTRIES = {
    "Kyrgyzstan",
    "United States",
}

ANONYMOUS_SENTENCE = (
    " The founding team currently lives and works remotely from [COUNTRY]."
)

failures: list[str] = []


def check(condition: bool, name: str, details: str = "") -> None:
    suffix = f" — {details}" if details else ""

    if condition:
        print(f"PASS | {name}{suffix}")
    else:
        print(f"FAIL | {name}{suffix}")
        failures.append(name + suffix)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen final design without "
            "calling any external API."
        )
    )

    parser.add_argument(
        "--allow-live-runtime",
        action="store_true",
        help=(
            "Allow the operational runtime switch to be true "
            "after freeze and OSF registration."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 76)
    print("FINAL MANIFEST PREFLIGHT")
    print("No APIs will be called and no money will be spent.")
    print("=" * 76)

    config_path = ROOT / "config/experiment.json"
    runtime_path = ROOT / "config/runtime.json"
    cases_path = ROOT / "scenarios/final_cases.json"

    check(config_path.exists(), "experiment.json exists")
    check(runtime_path.exists(), "runtime.json exists")
    check(cases_path.exists(), "final_cases.json exists")

    if failures:
        print("\nRESULT: FAIL")
        return 1

    cfg = load_json(config_path)
    runtime = load_json(runtime_path)
    cases = load_json(cases_path)

    check(isinstance(cfg, dict), "experiment.json is a JSON object")
    check(isinstance(runtime, dict), "runtime.json is a JSON object")
    check(isinstance(cases, list), "final_cases.json is a JSON list")

    if (
        not isinstance(cfg, dict)
        or not isinstance(runtime, dict)
        or not isinstance(cases, list)
    ):
        print("\nRESULT: FAIL")
        return 1

    runtime_enabled = runtime.get(
        "live_collection_enabled"
    )

    check(
        isinstance(
            runtime_enabled,
            bool,
        ),
        "runtime live switch is a boolean",
        f"actual={runtime_enabled!r}",
    )

    if args.allow_live_runtime:
        check(
            runtime_enabled is True,
            "runtime live switch is ON under --allow-live-runtime",
            f"actual={runtime_enabled!r}",
        )
    else:
        check(
            runtime_enabled is False,
            "runtime live switch remains OFF",
            f"actual={runtime_enabled!r}",
        )

    check(
        all(isinstance(case, dict) for case in cases),
        "every final case is a JSON object",
    )

    if not all(isinstance(case, dict) for case in cases):
        print("\nRESULT: FAIL")
        return 1

    models = cfg.get("models", [])
    case_ids = [case.get("id") for case in cases]
    mitigation_ids = cfg.get("mitigation_case_ids", [])

    print("\n--- Configuration checks ---")

    check(
        cfg.get("run_mode") == "dry_run",
        "run_mode remains dry_run",
        f"actual={cfg.get('run_mode')!r}",
    )

    check(
        cfg.get("repetitions") == 3,
        "repetitions = 3",
        f"actual={cfg.get('repetitions')!r}",
    )

    check(
        set(cfg.get("countries", [])) == EXPECTED_COUNTRIES,
        "countries are Kyrgyzstan and United States",
        f"actual={cfg.get('countries')!r}",
    )

    check(
        isinstance(cfg.get("seed"), int),
        "random seed is present",
        f"actual={cfg.get('seed')!r}",
    )

    check(
        cfg.get("collection_target_hours") == 24,
        "collection target = 24 hours",
        f"actual={cfg.get('collection_target_hours')!r}",
    )

    check(
        cfg.get("collection_hard_max_hours") == 72,
        "collection hard maximum = 72 hours",
        f"actual={cfg.get('collection_hard_max_hours')!r}",
    )

    check(
        isinstance(models, list) and len(models) == 3,
        "exactly 3 models are configured",
        f"actual={len(models) if isinstance(models, list) else 'not a list'}",
    )

    actual_models = {
        (
            model.get("provider"),
            model.get("model_id"),
        )
        for model in models
        if isinstance(model, dict)
    }

    check(
        actual_models == EXPECTED_MODELS,
        "selected model IDs are correct",
        f"actual={sorted(actual_models)}",
    )

    check(
        all(model.get("enabled") is False for model in models),
        "all final models remain disabled",
    )

    check(
        all(model.get("structured_output") is True for model in models),
        "structured output is enabled for every model",
    )

    check(
        all(model.get("max_output_tokens") == 16000 for model in models),
        "max_output_tokens = 16000 for every model",
    )

    print("\n--- Final-case checks ---")

    check(
        len(cases) == 8,
        "exactly 8 final cases are present",
        f"actual={len(cases)}",
    )

    check(
        all(
            {
                "id",
                "strength",
                "pitch_template",
            }.issubset(case)
            for case in cases
        ),
        "every case has id, strength, and pitch_template",
    )

    check(
        len(case_ids) == len(set(case_ids)),
        "all final case IDs are unique",
        f"IDs={case_ids}",
    )

    check(
        all(
            isinstance(case_id, str)
            and case_id
            and not case_id.upper().startswith("P")
            for case_id in case_ids
        ),
        "no pilot P-series cases are included",
        f"IDs={case_ids}",
    )

    allowed_strengths = {
        "strong",
        "borderline",
        "weak",
    }

    check(
        all(
            str(case.get("strength", "")).strip().lower()
            in allowed_strengths
            for case in cases
        ),
        "all strengths are Strong, Borderline, or Weak",
    )

    check(
        all(
            case["pitch_template"].count("[COUNTRY]") == 1
            for case in cases
        ),
        "every case has exactly one [COUNTRY] placeholder",
    )

    check(
        all(
            case["pitch_template"].count(ANONYMOUS_SENTENCE) == 1
            for case in cases
        ),
        "every case has exactly one removable country sentence",
    )

    check(
        len(mitigation_ids) == 8
        and len(set(mitigation_ids)) == 8,
        "exactly 8 unique mitigation cases are configured",
        f"actual={mitigation_ids}",
    )

    check(
        set(mitigation_ids) == set(case_ids),
        "mitigation is configured for all 8 final cases",
    )

    print("\n--- Planned-manifest checks ---")

    try:
        # This uses all three configured models even though they remain disabled.
        # It creates rows only in memory.
        rows = generate_rows(
            cfg=cfg,
            cases=cases,
            models=models,
        )

        check(
            True,
            "manifest generated safely in memory",
            f"rows={len(rows)}",
        )

    except Exception as exc:
        check(
            False,
            "manifest generated safely in memory",
            f"{type(exc).__name__}: {exc}",
        )

        print("\nRESULT: FAIL")
        return 1

    condition_counts = Counter(
        row["condition"]
        for row in rows
    )

    provider_counts = Counter(
        row["provider"]
        for row in rows
    )

    country_counts = Counter(
        row["country"]
        for row in rows
    )

    check(
        len(rows) == 360,
        "total planned runs = 360",
        f"actual={len(rows)}",
    )

    check(
        condition_counts["labelled"] == 144,
        "labelled runs = 144",
        f"actual={condition_counts['labelled']}",
    )

    check(
        condition_counts["anonymous"] == 72,
        "anonymous runs = 72",
        f"actual={condition_counts['anonymous']}",
    )

    check(
        condition_counts["mitigation"] == 144,
        "mitigation runs = 144",
        f"actual={condition_counts['mitigation']}",
    )

    check(
        all(
            provider_counts[provider] == 120
            for provider, model_id in EXPECTED_MODELS
        ),
        "each provider has exactly 120 planned runs",
        f"actual={dict(provider_counts)}",
    )

    expected_country_counts = Counter(
        {
            "Kyrgyzstan": 144,
            "United States": 144,
            "Anonymous": 72,
        }
    )

    check(
        country_counts == expected_country_counts,
        "country totals are balanced",
        f"actual={dict(country_counts)}",
    )

    run_ids = [
        row["run_id"]
        for row in rows
    ]

    check(
        len(run_ids) == len(set(run_ids)),
        "all 360 run IDs are unique",
    )

    check(
        all(
            "[COUNTRY]" not in row["user_prompt"]
            for row in rows
        ),
        "no unresolved [COUNTRY] placeholders remain",
    )

    check(
        all(
            row["scenario_id"] in set(case_ids)
            for row in rows
        ),
        "every planned row uses a final case",
    )

    check(
        all(
            (
                row["provider"],
                row["model_id"],
            )
            in EXPECTED_MODELS
            for row in rows
        ),
        "no unexpected models appear",
    )

    check(
        all(
            row["repetition"] in {
                1,
                2,
                3,
            }
            for row in rows
        ),
        "all repetitions are 1, 2, or 3",
    )

    check(
        all(
            row["prompt_hash"]
            == sha256_text(
                row["system_prompt"]
                + "\n"
                + row["user_prompt"]
            )
            for row in rows
        ),
        "all prompt hashes are correct",
    )

    anonymous_errors = [
        row["run_id"]
        for row in rows
        if row["condition"] == "anonymous"
        and (
            row["country"] != "Anonymous"
            or
            "The founding team currently lives and works remotely from"
            in row["user_prompt"]
        )
    ]

    check(
        not anonymous_errors,
        "anonymous prompts remove the country sentence",
        f"bad rows={len(anonymous_errors)}",
    )

    labelled_errors = [
        row["run_id"]
        for row in rows
        if row["condition"] in {
            "labelled",
            "mitigation",
        }
        and (
            row["country"] not in EXPECTED_COUNTRIES
            or
            (
                "The founding team currently lives and works remotely from "
                f"{'the United States' if row['country'] == 'United States' else row['country']}."
            )
            not in row["user_prompt"]
        )
    ]

    check(
        not labelled_errors,
        "labelled and mitigation prompts contain the assigned country",
        f"bad rows={len(labelled_errors)}",
    )

    pair_groups: dict[tuple, list[dict]] = defaultdict(list)

    for row in rows:
        if row["condition"] not in {
            "labelled",
            "mitigation",
        }:
            continue

        key = (
            row["condition"],
            row["provider"],
            row["model_id"],
            row["scenario_id"],
            row["repetition"],
        )

        pair_groups[key].append(row)

    bad_pairs = 0

    for pair in pair_groups.values():
        pair_countries = {
            row["country"]
            for row in pair
        }

        normalized_prompts = {
            row["user_prompt"]
            .replace(
                "Kyrgyzstan",
                "[COUNTRY]",
            )
            .replace(
                "the United States",
                "[COUNTRY]",
            )
            for row in pair
        }

        if (
            len(pair) != 2
            or pair_countries != EXPECTED_COUNTRIES
            or len(normalized_prompts) != 1
        ):
            bad_pairs += 1

    check(
        bad_pairs == 0,
        "all Kyrgyzstan-US pairs differ only by country",
        f"bad pairs={bad_pairs}",
    )

    print("\n--- Git-safety checks ---")

    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        tracked_files = result.stdout.splitlines()

        check(
            ".env" not in tracked_files,
            ".env is not tracked by Git",
        )

        check(
            not any(
                path == ".venv"
                or path.startswith(".venv/")
                for path in tracked_files
            ),
            ".venv is not tracked by Git",
        )

    except Exception as exc:
        print(
            "WARN | Git-safety check could not run"
            f" — {exc}"
        )

    print("\n--- Final summary ---")

    print(
        f"Total runs:            {len(rows)}"
    )
    print(
        f"Labelled:              {condition_counts['labelled']}"
    )
    print(
        f"Anonymous:             {condition_counts['anonymous']}"
    )
    print(
        f"Mitigation:            {condition_counts['mitigation']}"
    )
    print(
        f"Kyrgyzstan rows:       {country_counts['Kyrgyzstan']}"
    )
    print(
        f"United States rows:    {country_counts['United States']}"
    )
    print(
        f"Anonymous rows:        {country_counts['Anonymous']}"
    )
    print(
        "Duplicate run IDs:     "
        f"{len(run_ids) - len(set(run_ids))}"
    )
    print(
        f"Bad matched pairs:     {bad_pairs}"
    )
    print(
        "Final models enabled:  NO"
    )

    if failures:
        print(
            f"\nRESULT: FAIL "
            f"({len(failures)} failed checks)"
        )
        return 1

    print("\nRESULT: PASS")
    print(
        "No API calls were made. "
        "No manifest file was written."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
