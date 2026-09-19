from __future__ import annotations

from pathlib import Path

from common import ROOT


EXCLUDED = {
    "config/runtime.json",
    "docs/FREEZE_MANIFEST.json",
    "docs/POSTCOLLECTION_MANIFEST.json",
    "docs/OSF_REGISTRATION_CONFIRMED.txt",
}

PATTERNS = [
    "config/*.json",
    "prompts/*",
    "scenarios/*.json",
    "providers/*.py",
    "analysis/*.py",
    "scripts/*.py",
    "tests/*.py",
    "requirements*.txt",
    "docs/**/*.md",
    "docs/**/*.txt",
    "docs/**/*.json",
]


def scientific_files(
    root: Path = ROOT,
) -> list[Path]:
    files: set[Path] = set()

    for pattern in PATTERNS:
        for path in root.glob(
            pattern
        ):
            if not path.is_file():
                continue

            relative = path.relative_to(
                root
            ).as_posix()

            if relative in EXCLUDED:
                continue

            if "__pycache__" in path.parts:
                continue

            files.add(path)

    return sorted(
        files,
        key=lambda p: (
            p.relative_to(root)
            .as_posix()
        ),
    )
