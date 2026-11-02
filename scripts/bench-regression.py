#!/usr/bin/env python3
"""Fail when a benchmark run regresses against a committed baseline.

The comparison is deliberately narrow: p99 write latency and ingest wall time.
Those are the two numbers the storage design exists to protect. Throughput is
derived from ingest time, so gating ingest time covers both.

Exit codes:
  0  within budget
  1  regression beyond the budget
  2  cannot compare (missing field, missing file) — treated as failure, because a
     gate that silently passes when it cannot read the data is not a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--p99-budget",
        type=float,
        default=0.20,
        help="fractional regression in p99 that is tolerated (0.20 = +20%%)",
    )
    parser.add_argument(
        "--ingest-budget",
        type=float,
        default=0.25,
        help="fractional regression in ingest wall time that is tolerated",
    )
    args = parser.parse_args()

    for label, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        if not path.exists():
            print(f"FAIL: {label} not found: {path}")
            return 2

    try:
        base = load(args.baseline)
        cand = load(args.candidate)
        base_p99 = float(base["latency_us"]["p99"])
        cand_p99 = float(cand["latency_us"]["p99"])
        base_ingest = float(base["ingest_ms"])
        cand_ingest = float(cand["ingest_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"FAIL: cannot compare: {exc}")
        return 2

    # A machine difference would make any threshold meaningless; refuse rather
    # than report a number with no basis.
    for field in ("records", "batch", "dim", "durability"):
        if base.get(field) != cand.get(field):
            print(
                f"FAIL: not comparable — {field} differs "
                f"({base.get(field)!r} vs {cand.get(field)!r})"
            )
            return 2

    failures = []

    p99_ratio = (cand_p99 / base_p99) - 1.0 if base_p99 else 0.0
    verdict = "OK" if p99_ratio <= args.p99_budget else "FAIL"
    print(
        f"{verdict:4} p99 {base_p99 / 1000:8.2f} ms -> {cand_p99 / 1000:8.2f} ms "
        f"({p99_ratio:+.1%}, budget +{args.p99_budget:.0%})"
    )
    if p99_ratio > args.p99_budget:
        failures.append(f"p99 regressed {p99_ratio:+.1%}")

    ingest_ratio = (cand_ingest / base_ingest) - 1.0 if base_ingest else 0.0
    verdict = "OK" if ingest_ratio <= args.ingest_budget else "FAIL"
    print(
        f"{verdict:4} ingest {base_ingest / 1000:7.2f} s -> {cand_ingest / 1000:7.2f} s "
        f"({ingest_ratio:+.1%}, budget +{args.ingest_budget:.0%})"
    )
    if ingest_ratio > args.ingest_budget:
        failures.append(f"ingest regressed {ingest_ratio:+.1%}")

    if failures:
        print("REGRESSION: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
