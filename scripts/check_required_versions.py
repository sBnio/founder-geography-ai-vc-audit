from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "requirements-lock.txt"


def parse_lock(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Lock file not found: {path}"
        )

    requirements = []

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "==" not in line:
            raise ValueError(
                "Every lock entry must use exact == pinning: "
                f"{line}"
            )

        name, version = line.split("==", 1)

        name = name.strip()
        version = version.strip()

        if not name or not version:
            raise ValueError(
                f"Invalid lock entry: {line}"
            )

        requirements.append(
            (name, version)
        )

    if not requirements:
        raise ValueError(
            "requirements-lock.txt is empty."
        )

    return requirements


def main() -> int:
    print("=" * 76)
    print("EXACT DEPENDENCY VERSION CHECK")
    print("=" * 76)

    try:
        requirements = parse_lock(
            LOCK_PATH
        )
    except Exception as exc:
        print(f"FAIL | {exc}")
        return 1

    failures = []

    for package_name, expected in requirements:
        try:
            actual = metadata.version(
                package_name
            )
        except metadata.PackageNotFoundError:
            actual = None

        passed = actual == expected

        print(
            f"{'PASS' if passed else 'FAIL'} | "
            f"{package_name}: "
            f"{actual!r} "
            f"(expected {expected!r})"
        )

        if not passed:
            failures.append(
                (
                    package_name,
                    actual,
                    expected,
                )
            )

    print()
    print(
        "Python:",
        sys.version.split()[0],
    )

    if failures:
        print(
            f"\nRESULT: FAIL "
            f"({len(failures)} mismatches)"
        )
        return 1

    print(
        f"\nRESULT: PASS "
        f"({len(requirements)} exact package pins)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
