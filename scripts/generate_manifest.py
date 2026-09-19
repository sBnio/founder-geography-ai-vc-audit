import argparse
import hashlib
import json
import random
import uuid
from pathlib import Path

from common import ROOT, load_json, read_text, sha256_text


def build_prompt(template: str, pitch: str) -> str:
    return template.replace("{PITCH}", pitch)


def synthetic_models(count: int) -> list[dict]:
    if count < 1:
        raise ValueError("synthetic model count must be at least 1")
    return [
        {
            "provider": "dry_run",
            "model_id": f"synthetic-model-{i+1}",
            "temperature": 0.2,
            "max_output_tokens": 1200,
        }
        for i in range(count)
    ]


def generate_rows(cfg: dict, cases: list[dict], models: list[dict]) -> list[dict]:
    system = read_text(ROOT / "prompts/system.txt")
    main_prompt = read_text(ROOT / "prompts/main.txt")
    mitigation = read_text(ROOT / "prompts/mitigation.txt")
    rows: list[dict] = []

    for model in models:
        for case in cases:
            for repetition in range(1, cfg["repetitions"] + 1):
                for country in cfg["countries"]:
                    country_text = "the United States" if country == "United States" else country
                    pitch = case["pitch_template"].replace("[COUNTRY]", country_text)
                    user_prompt = build_prompt(main_prompt, pitch)
                    key = f'labelled|{model["provider"]}|{model["model_id"]}|{case["id"]}|{country}|{repetition}'
                    rows.append(
                        {
                            "run_id": str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
                            "condition": "labelled",
                            "scenario_id": case["id"],
                            "strength": case["strength"],
                            "country": country,
                            "provider": model["provider"],
                            "model_id": model["model_id"],
                            "repetition": repetition,
                            "system_prompt": system,
                            "user_prompt": user_prompt,
                            "prompt_hash": sha256_text(system + "\n" + user_prompt),
                        }
                    )

                anonymous_sentence = " The founding team currently lives and works remotely from [COUNTRY]."
                if anonymous_sentence not in case["pitch_template"]:
                    raise ValueError(f'{case["id"]}: anonymous sentence not found exactly once')
                pitch = case["pitch_template"].replace(anonymous_sentence, "")
                user_prompt = build_prompt(main_prompt, pitch)
                key = f'anonymous|{model["provider"]}|{model["model_id"]}|{case["id"]}|{repetition}'
                rows.append(
                    {
                        "run_id": str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
                        "condition": "anonymous",
                        "scenario_id": case["id"],
                        "strength": case["strength"],
                        "country": "Anonymous",
                        "provider": model["provider"],
                        "model_id": model["model_id"],
                        "repetition": repetition,
                        "system_prompt": system,
                        "user_prompt": user_prompt,
                        "prompt_hash": sha256_text(system + "\n" + user_prompt),
                    }
                )

                if case["id"] in cfg["mitigation_case_ids"]:
                    for country in cfg["countries"]:
                        country_text = "the United States" if country == "United States" else country
                        pitch = case["pitch_template"].replace("[COUNTRY]", country_text)
                        user_prompt = build_prompt(main_prompt, pitch) + "\n\n" + mitigation
                        key = f'mitigation|{model["provider"]}|{model["model_id"]}|{case["id"]}|{country}|{repetition}'
                        rows.append(
                            {
                                "run_id": str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
                                "condition": "mitigation",
                                "scenario_id": case["id"],
                                "strength": case["strength"],
                                "country": country,
                                "provider": model["provider"],
                                "model_id": model["model_id"],
                                "repetition": repetition,
                                "system_prompt": system,
                                "user_prompt": user_prompt,
                                "prompt_hash": sha256_text(system + "\n" + user_prompt),
                            }
                        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a randomized, deterministic run manifest.")
    parser.add_argument("--mode", choices=["config", "synthetic"], default="config")
    parser.add_argument("--synthetic-models", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/run_manifest.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(ROOT / "config/experiment.json")
    cases = load_json(ROOT / "scenarios/final_cases.json")
    if args.mode == "synthetic":
        models = synthetic_models(args.synthetic_models)
    else:
        # Final scientific model identities remain frozen in
        # experiment.json with enabled=false. Operational permission
        # to collect lives only in config/runtime.json.
        models = list(cfg["models"])

        if not models:
            raise RuntimeError(
                "No final models are configured in experiment.json"
            )

        if any(
            "FILL_" in m.get("model_id", "")
            for m in models
        ):
            raise RuntimeError(
                "Replace all placeholder model IDs "
                "before generating a final manifest"
            )
    rows = generate_rows(cfg, cases, models)
    random.Random(cfg["seed"]).shuffle(rows)
    if len({row["run_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate run IDs detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_sha256 = hashlib.sha256(
        args.output.read_bytes()
    ).hexdigest()

    sidecar_path = Path(
        str(args.output) + ".sha256"
    )

    sidecar_path.write_text(
        f"{manifest_sha256}  {args.output.name}\n",
        encoding="utf-8",
    )

    print(
        {
            "runs": len(rows),
            "models": len(models),
            "mode": args.mode,
            "output": str(args.output),
            "sha256": manifest_sha256,
            "sidecar": str(sidecar_path),
        }
    )


if __name__ == "__main__":
    main()
