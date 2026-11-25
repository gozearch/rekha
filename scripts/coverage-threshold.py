#!/usr/bin/env python3
"""Fail when line coverage in an lcov report is below a floor.

Codecov's status check is the public number, but it only runs when the upload
has a token. This is the same measurement locally, with no account: `LF`/`LH`
records summed exactly as Codecov sums them, so the gate a contributor runs
before pushing and the gate the PR shows agree.

Usage:
    python3 scripts/coverage-threshold.py lcov.info [85]
"""

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = pathlib.Path(sys.argv[1])
    floor = float(sys.argv[2]) if len(sys.argv) > 2 else 85.0

    lines_found = 0
    lines_hit = 0
    for line in path.read_text().splitlines():
        if line.startswith("LF:"):
            lines_found += int(line[3:])
        elif line.startswith("LH:"):
            lines_hit += int(line[3:])

    if lines_found == 0:
        print(f"no coverage records in {path}", file=sys.stderr)
        return 2

    percent = 100.0 * lines_hit / lines_found
    print(f"line coverage: {lines_hit}/{lines_found} = {percent:.2f}% (floor {floor:.2f}%)")
    return 0 if percent + 1e-9 >= floor else 1


if __name__ == "__main__":
    raise SystemExit(main())
