from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
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


def git_output(
    args: list[str],
) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
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

    if args.out.exists():
        raise SystemExit(
            "ERROR: Refusing to overwrite an existing "
            "scientific freeze manifest."
        )

    raw_path = (
        ROOT
        / "data"
        / "raw"
        / "responses.jsonl"
    )

    if (
        raw_path.exists()
        and raw_path.stat().st_size > 0
    ):
        raise SystemExit(
            "ERROR: Final raw responses already exist. "
            "Scientific freeze must precede collection."
        )

    status = git_output(
        [
            "status",
            "--porcelain",
        ]
    )

    if status:
        raise SystemExit(
            "ERROR: Repository is not clean. "
            "Commit/archive the approved pre-collection "
            "state before creating the scientific freeze."
        )

    files = scientific_files()

    if not files:
        raise SystemExit(
            "ERROR: Scientific freeze scope is empty."
        )

    records = []

    for path in files:
        records.append(
            {
                "path": (
                    path.relative_to(
                        ROOT
                    ).as_posix()
                ),
                "size_bytes": (
                    path.stat().st_size
                ),
                "sha256": (
                    sha256_file(path)
                ),
            }
        )

    manifest = {
        "freeze_format_version": 1,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "git_commit": git_output(
            [
                "rev-parse",
                "HEAD",
            ]
        ),
        "algorithm": "SHA-256",
        "runtime_json_excluded": True,
        "file_count": len(records),
        "files": records,
    }

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.out.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("SCIENTIFIC FREEZE CREATED")
    print("=" * 76)
    print("Files:", len(records))
    print("Output:", args.out)
    print(
        "config/runtime.json included: NO"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
