"""Fail when raw branch coverage is below the requested threshold."""

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from typing import Any


def _coverage_json() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "coverage", "json", "-o", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _branch_totals(report: dict[str, Any]) -> tuple[int, int]:
    if report.get("meta", {}).get("branch_coverage") is not True:
        raise ValueError("coverage data was not collected with branch measurement enabled")

    totals = report.get("totals", {})
    covered = int(totals.get("covered_branches", 0))
    branches = int(totals.get("num_branches", 0))
    if branches <= 0:
        raise ValueError("coverage data contains no measurable branches")
    return covered, branches


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum", type=float, default=85.0)
    args = parser.parse_args(argv)

    try:
        covered, branches = _branch_totals(_coverage_json())
    except (json.JSONDecodeError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Unable to verify branch coverage: {exc}", file=sys.stderr)
        return 2

    percentage = covered * 100.0 / branches
    print(
        f"Raw branch coverage: {percentage:.2f}% "
        f"({covered}/{branches}; required {args.minimum:.2f}%)"
    )
    return 0 if percentage >= args.minimum else 1


if __name__ == "__main__":
    raise SystemExit(main())
