import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dry_manifest_unique(tmp_path):
    out = tmp_path / "test_manifest.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_manifest.py"),
            "--mode",
            "synthetic",
            "--synthetic-models",
            "1",
            "--output",
            str(out),
        ],
        check=True,
        cwd=ROOT / "scripts",
    )

    rows = [
        json.loads(line)
        for line in out.read_text().splitlines()
    ]

    assert len(rows) == 120
    assert len({row["run_id"] for row in rows}) == len(rows)
