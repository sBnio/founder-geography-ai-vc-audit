from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def run_text(
    command: list[str],
) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    return result.stdout.strip()


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    return bool(
        result.stdout.strip()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a privacy-conscious "
            "precollection software environment."
        )
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "PRECOLLECTION_ENVIRONMENT.json"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    lock_path = (
        ROOT
        / "requirements-lock.txt"
    )

    experiment_path = (
        ROOT
        / "config"
        / "experiment.json"
    )

    if not lock_path.exists():
        raise SystemExit(
            "ERROR: requirements-lock.txt "
            "does not exist."
        )

    if not experiment_path.exists():
        raise SystemExit(
            "ERROR: config/experiment.json "
            "does not exist."
        )

    freeze_text = run_text(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ]
    )

    if freeze_text is None:
        raise SystemExit(
            "ERROR: Could not run pip freeze."
        )

    environment = {
        "captured_at_utc": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),
        "python": {
            "version": (
                platform.python_version()
            ),
            "implementation": (
                platform.python_implementation()
            ),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": {
            "requirements_lock_sha256": (
                sha256_file(
                    lock_path
                )
            ),
            "pip_freeze": (
                freeze_text.splitlines()
            ),
        },
        "scientific_config": {
            "experiment_json_sha256": (
                sha256_file(
                    experiment_path
                )
            ),
        },
        "git": {
            "commit": run_text(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ]
            ),
            "dirty": git_dirty(),
        },
        "privacy_note": (
            "Environment variables, API keys, "
            ".env contents, usernames, and "
            "absolute executable paths are "
            "intentionally not captured."
        ),
    }

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.out.write_text(
        json.dumps(
            environment,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("ENVIRONMENT CAPTURE")
    print("=" * 76)
    print(f"Output: {args.out}")
    print(
        "Python:",
        environment[
            "python"
        ]["version"],
    )
    print(
        "Platform:",
        environment[
            "platform"
        ]["system"],
        environment[
            "platform"
        ]["machine"],
    )
    print(
        "Locked packages:",
        len(
            environment[
                "dependencies"
            ]["pip_freeze"]
        ),
    )
    print(
        "Git dirty:",
        environment[
            "git"
        ]["dirty"],
    )
    print(
        "Secrets/environment variables captured: NO"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
