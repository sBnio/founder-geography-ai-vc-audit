from __future__ import annotations

import re
import subprocess
from pathlib import Path

from common import ROOT


SECRET_PATTERNS = {
    "OpenAI/Anthropic-style secret key": re.compile(
        rb"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"
    ),
    "Google-style API key": re.compile(
        rb"AIza[0-9A-Za-z_-]{20,}"
    ),
}

FORBIDDEN_TRACKED = (
    ".env",
    ".venv/",
    "data/raw/",
    "data/pilot/",
    "data/coding/PRIVATE_",
)


def git(
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        check=check,
    )


def main() -> int:
    print("=" * 76)
    print("SECURITY AUDIT")
    print("=" * 76)

    failures = []

    tracked_result = git(
        "ls-files"
    )

    tracked = (
        tracked_result.stdout
        .decode(
            "utf-8",
            errors="replace",
        )
        .splitlines()
    )

    for path in tracked:
        if (
            path == ".env"
            or path.startswith(
                ".venv/"
            )
            or path.startswith(
                "data/raw/"
            )
            or path.startswith(
                "data/pilot/"
            )
            or path.startswith(
                "data/coding/PRIVATE_"
            )
        ):
            failures.append(
                f"forbidden tracked path: {path}"
            )

    print(
        "PASS" if not any(
            item.startswith(
                "forbidden tracked path:"
            )
            for item in failures
        ) else "FAIL",
        "| private/raw paths not tracked",
    )

    current_secret_hits = []

    for relative in tracked:
        path = ROOT / relative

        if (
            not path.exists()
            or not path.is_file()
        ):
            continue

        try:
            data = path.read_bytes()
        except OSError:
            continue

        if b"\x00" in data[:4096]:
            continue

        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                current_secret_hits.append(
                    f"{relative} ({label})"
                )

    if current_secret_hits:
        failures.extend(
            "possible current secret: "
            + hit
            for hit in current_secret_hits
        )
        print(
            "FAIL | possible API-key material "
            "found in tracked current files"
        )
    else:
        print(
            "PASS | no API-key-shaped secrets "
            "found in tracked current files"
        )

    env_history = git(
        "log",
        "--all",
        "--format=%H",
        "--",
        ".env",
    ).stdout.decode(
        "utf-8",
        errors="replace",
    ).strip()

    if env_history:
        failures.append(
            ".env appears in Git history"
        )
        print(
            "FAIL | .env appears in Git history"
        )
    else:
        print(
            "PASS | .env absent from Git history"
        )

    commits = (
        git(
            "rev-list",
            "--all",
        )
        .stdout.decode(
            "utf-8",
            errors="replace",
        )
        .splitlines()
    )

    historical_secret_hits = []

    grep_patterns = {
        "OpenAI/Anthropic-style secret key":
            r"sk-(proj-|ant-)?[A-Za-z0-9_-]{20,}",
        "Google-style API key":
            r"AIza[0-9A-Za-z_-]{20,}",
    }

    for commit in commits:
        for label, pattern in grep_patterns.items():
            result = git(
                "grep",
                "-I",
                "-l",
                "-E",
                pattern,
                commit,
                check=False,
            )

            if result.returncode == 0:
                paths = (
                    result.stdout.decode(
                        "utf-8",
                        errors="replace",
                    )
                    .splitlines()
                )

                historical_secret_hits.extend(
                    f"{commit[:12]}:{path} ({label})"
                    for path in paths
                )

    if historical_secret_hits:
        failures.extend(
            "possible historical secret: "
            + hit
            for hit in historical_secret_hits
        )

        print(
            "FAIL | possible API-key material "
            "found in Git history"
        )
    else:
        print(
            "PASS | no API-key-shaped secrets "
            "found in Git history"
        )

    gitignore = (
        ROOT / ".gitignore"
    ).read_text(
        encoding="utf-8"
    )

    required_ignore_fragments = [
        ".env",
        ".venv/",
        "data/raw/",
        "data/pilot/",
        "data/coding/PRIVATE_",
    ]

    missing_ignores = [
        item
        for item in required_ignore_fragments
        if item not in gitignore
    ]

    if missing_ignores:
        failures.append(
            "missing .gitignore protections: "
            f"{missing_ignores}"
        )
        print(
            "FAIL | required .gitignore "
            "protections incomplete"
        )
    else:
        print(
            "PASS | required .gitignore "
            "protections present"
        )

    if failures:
        print()
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

    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
