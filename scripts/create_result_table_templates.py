from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "table_templates"


TEMPLATES = {
    "table_1_models_and_completion.csv": [
        "provider",
        "model_id",
        "configured_reasoning_setting",
        "planned_runs",
        "successful_runs",
        "failed_runs",
        "runs_requiring_retry",
        "mean_input_tokens",
        "mean_visible_output_tokens",
        "mean_reasoning_tokens",
        "mean_latency_seconds",
        "estimated_cost_usd",
        "actual_charge_usd",
        "notes",
    ],

    "table_2_primary_country_effect.csv": [
        "n_valid_labelled_outputs",
        "n_matched_scenario_model_contrasts",
        "n_scenarios",
        "mean_gap_kyrgyzstan_minus_us",
        "median_gap_kyrgyzstan_minus_us",
        "standard_deviation",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "exact_sign_flip_p",
        "equivalence_margin_points",
        "tost_p_lower",
        "tost_p_upper",
        "tost_max_p",
        "equivalent_within_margin",
        "interpretation",
    ],

    "table_3_model_specific_effects.csv": [
        "provider",
        "model_id",
        "n_matched_pairs",
        "mean_gap_kyrgyzstan_minus_us",
        "median_gap_kyrgyzstan_minus_us",
        "standard_deviation",
        "notes",
    ],

    "table_4_secondary_outcomes.csv": [
        "outcome",
        "condition",
        "n_matched_pairs",
        "mean_gap_kyrgyzstan_minus_us",
        "median_gap_kyrgyzstan_minus_us",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "notes",
    ],

    "table_5_qualitative_coding.csv": [
        "coding_category",
        "category_definition",
        "total_count",
        "total_proportion",
        "kyrgyzstan_count",
        "united_states_count",
        "anonymous_count",
        "mitigation_count",
        "coder_1_count",
        "coder_2_count",
        "agreement_statistic",
        "notes",
    ],

    "table_6_robustness_and_controls.csv": [
        "analysis_name",
        "sample_or_condition",
        "estimate",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "p_value",
        "conclusion",
        "notes",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create blank result-table templates before "
            "final data collection."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing blank template files.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    skipped = []

    for filename, columns in TEMPLATES.items():
        path = args.output_dir / filename

        if path.exists() and not args.force:
            skipped.append(path)
            continue

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)

        created.append(path)

    readme = args.output_dir / "README.txt"

    if not readme.exists() or args.force:
        readme.write_text(
            "These table structures were created before final "
            "country-labelled data collection.\n\n"
            "They contain headers only and no research results.\n"
            "Do not enter synthetic results into these tables.\n"
            "Final values must be copied from the frozen analysis "
            "pipeline without changing the preregistered rules.\n",
            encoding="utf-8",
        )

        created.append(readme)

    print("=" * 76)
    print("FINAL RESULT-TABLE TEMPLATES")
    print("=" * 76)

    print(f"\nOutput directory:\n{args.output_dir}")

    if created:
        print("\nCreated:")
        for path in created:
            print(" -", path.relative_to(ROOT))

    if skipped:
        print("\nAlready existed and were not replaced:")
        for path in skipped:
            print(" -", path.relative_to(ROOT))

    print("\nNo API calls were made.")
    print("No research results were entered.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
