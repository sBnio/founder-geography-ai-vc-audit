from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def file_record(path: Path) -> dict | None:
    if not path.exists():
        return None

    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Back up and hash final raw collection data "
            "without modifying the original."
        )
    )

    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data/raw/responses.jsonl",
    )

    parser.add_argument(
        "--processed",
        type=Path,
        default=ROOT / "data/processed/responses.csv",
    )

    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=ROOT / "data/processed/run_manifest.jsonl",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/POSTCOLLECTION_MANIFEST.json",
    )

    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=ROOT / "data/raw/backup",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.out.exists():
        raise SystemExit(
            "ERROR: Refusing to overwrite existing "
            "POSTCOLLECTION_MANIFEST.json."
        )

    if (
        not args.raw.exists()
        or args.raw.stat().st_size == 0
    ):
        raise SystemExit(
            "ERROR: Raw response log is missing or empty."
        )

    records = []

    for line_number, line in enumerate(
        args.raw.read_text(
            encoding="utf-8"
        ).splitlines(),
        1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "ERROR: Raw JSONL contains invalid JSON "
                f"at line {line_number}: {exc}"
            )

        records.append(record)

    if not records:
        raise SystemExit(
            "ERROR: Raw JSONL contains no attempt records."
        )

    run_ids = {
        str(record.get("run_id"))
        for record in records
        if record.get("run_id")
    }

    successful_run_ids = {
        str(record.get("run_id"))
        for record in records
        if (
            record.get("run_id")
            and record.get("status") == "success"
        )
    }

    error_attempts = sum(
        record.get("status") == "error"
        for record in records
    )

    timestamps = sorted(
        str(record["timestamp_utc"])
        for record in records
        if record.get("timestamp_utc")
    )

    now = datetime.now(
        timezone.utc
    )

    stamp = now.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    args.backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup = (
        args.backup_dir
        / f"responses_{stamp}.jsonl"
    )

    if backup.exists():
        raise SystemExit(
            f"ERROR: Backup already exists: {backup}"
        )

    shutil.copy2(
        args.raw,
        backup,
    )

    os.chmod(
        backup,
        0o444,
    )

    raw_hash = sha256_file(
        args.raw
    )

    backup_hash = sha256_file(
        backup
    )

    if raw_hash != backup_hash:
        raise SystemExit(
            "ERROR: Read-only backup hash does not "
            "match original raw data."
        )

    manifest_sidecar = Path(
        str(args.run_manifest) + ".sha256"
    )

    files = {}

    paths = {
        "raw_responses": args.raw,
        "raw_read_only_backup": backup,
        "processed_responses": args.processed,
        "run_manifest": args.run_manifest,
        "run_manifest_sha256_sidecar": manifest_sidecar,
        "scientific_freeze_manifest": (
            ROOT / "docs/FREEZE_MANIFEST.json"
        ),
        "precollection_environment": (
            ROOT / "docs/PRECOLLECTION_ENVIRONMENT.json"
        ),
    }

    for name, path in paths.items():
        record = file_record(path)

        if record is not None:
            files[name] = record

    output = {
        "format_version": 1,
        "created_at_utc": now.isoformat(),
        "raw_attempt_records": len(records),
        "unique_planned_run_ids_seen": len(run_ids),
        "unique_successful_run_ids": len(
            successful_run_ids
        ),
        "error_attempt_records": int(
            error_attempts
        ),
        "first_attempt_timestamp_utc": (
            timestamps[0]
            if timestamps
            else None
        ),
        "last_attempt_timestamp_utc": (
            timestamps[-1]
            if timestamps
            else None
        ),
        "raw_backup_verified_identical": True,
        "files": files,
    }

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.out.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("POSTCOLLECTION INTEGRITY MANIFEST")
    print("=" * 76)
    print(
        "Attempt records:",
        len(records),
    )
    print(
        "Unique run IDs:",
        len(run_ids),
    )
    print(
        "Successful run IDs:",
        len(successful_run_ids),
    )
    print(
        "Error attempts:",
        error_attempts,
    )
    print(
        "Raw/backup SHA match: YES"
    )
    print(
        "Read-only backup:",
        backup,
    )
    print(
        "Manifest:",
        args.out,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
