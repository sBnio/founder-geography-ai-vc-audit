from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common import ROOT
from freeze_scope import scientific_files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "FREEZE_MANIFEST.json"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.manifest.exists():
        print(
            "FAIL | Scientific freeze "
            "manifest does not exist."
        )
        return 1

    manifest = json.loads(
        args.manifest.read_text(
            encoding="utf-8"
        )
    )

    records = manifest.get(
        "files",
        [],
    )

    expected_paths = {
        record["path"]
        for record in records
    }

    current_paths = {
        path.relative_to(
            ROOT
        ).as_posix()
        for path in scientific_files()
    }

    failures = []

    if (
        "config/runtime.json"
        in expected_paths
    ):
        failures.append(
            "runtime.json was incorrectly "
            "included in scientific freeze"
        )

    missing_scope = (
        expected_paths
        - current_paths
    )

    extra_scope = (
        current_paths
        - expected_paths
    )

    if missing_scope:
        failures.append(
            "missing scientific files: "
            f"{sorted(missing_scope)}"
        )

    if extra_scope:
        failures.append(
            "new/unfrozen scientific files: "
            f"{sorted(extra_scope)}"
        )

    for record in records:
        path = (
            ROOT
            / record["path"]
        )

        if not path.exists():
            failures.append(
                f"missing: {record['path']}"
            )
            continue

        actual = sha256_file(
            path
        )

        if actual != record["sha256"]:
            failures.append(
                f"hash mismatch: "
                f"{record['path']}"
            )

    print("=" * 76)
    print("SCIENTIFIC FREEZE VERIFICATION")
    print("=" * 76)
    print(
        "Frozen files:",
        len(records),
    )

    if failures:
        for failure in failures:
            print(
                "FAIL |",
                failure,
            )

        print(
            f"\nRESULT: FAIL "
            f"({len(failures)} problems)"
        )
        return 1

    print(
        "PASS | All scientific hashes match."
    )
    print(
        "PASS | runtime.json remains outside "
        "the scientific freeze."
    )
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
