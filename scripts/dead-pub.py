#!/usr/bin/env python3
"""Fail when a `pub` item in a library crate is referenced nowhere.

`dead_code = "deny"` cannot see this: rustc only reports an item as dead when
it is unreachable *from the crate root*, and a `pub` item in a library crate is
by definition reachable from it. That is how eight unused functions and one
unused type survived a deny-level lint. This script closes the hole from the
outside, using the only signal available without nightly rustdoc JSON: the item
is defined and its name never appears anywhere else in the workspace.

It under-detects rather than over-detects. A name shared by two items (`new`,
`len`) is never reported, and a re-export counts as a reference. That is the
right trade for a gate that runs on every PR: a false positive here blocks
merges, a false negative costs one dead function.

Usage:
    python3 scripts/dead-pub.py [--list]

`--list` prints every item and its reference count instead of failing.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "scripts" / "dead-pub-allow.txt"

DEF_PATTERN = re.compile(
    r"^\s*pub\s+(?:async\s+)?(?:unsafe\s+)?(?:fn|struct|enum|trait|union|type|const|static)\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)

# Where a reference may legitimately live. `tests/` and `bench/` included on
# purpose: an item used only by an integration test is not dead, it is a test
# seam, and the allowlist is where that decision is recorded.
SOURCE_GLOBS = [
    "crates/*/src/**/*.rs",
    "crates/*/tests/**/*.rs",
    "crates/*/benches/**/*.rs",
    "bench/**/*.rs",
    "tests/**/*.rs",
]

DEF_GLOBS = ["crates/*/src/**/*.rs"]

# A `#[cfg(test)] mod tests` block at the bottom of a file defines items that
# are only reachable when testing. They are not production API and must not be
# reported, so the definition scan stops there.
TEST_CUT = "#[cfg(test)]"


def definitions() -> dict[str, pathlib.Path]:
    found: dict[str, pathlib.Path] = {}
    for pattern in DEF_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            text = path.read_text(encoding="utf-8", errors="ignore")
            cut = text.find(TEST_CUT)
            production = text[:cut] if cut != -1 else text
            for match in DEF_PATTERN.finditer(production):
                # First definition wins; a later same-name item is counted by
                # the shared reference total either way.
                found.setdefault(match.group(1), path)
    return found


def references(name: str, corpus: list[tuple[pathlib.Path, str]]) -> int:
    expression = re.compile(r"\b" + re.escape(name) + r"\b")
    return sum(len(expression.findall(text)) for _, text in corpus)


def allowlisted() -> set[str]:
    if not ALLOWLIST.exists():
        return set()
    allowed: set[str] = set()
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            allowed.add(line.split()[0])
    return allowed


def main(argv: list[str]) -> int:
    corpus: list[tuple[pathlib.Path, str]] = []
    for pattern in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            corpus.append(
                (path, path.read_text(encoding="utf-8", errors="ignore"))
            )

    allowed = allowlisted()
    defs = definitions()
    rows = []
    for name, path in sorted(defs.items()):
        if name in allowed:
            continue
        rows.append((references(name, corpus), name, path))

    if "--list" in argv:
        for count, name, path in sorted(rows):
            print(f"{count:5}  {name:48} {path.relative_to(ROOT)}")
        return 0

    dead = [(count, name, path) for count, name, path in rows if count <= 1]
    if not dead:
        print(f"dead-pub: {len(rows)} public items, all referenced")
        return 0

    print(
        "dead-pub: these `pub` items are defined and referenced nowhere in\n"
        "src/, tests/ or bench/. Delete them, or record the reason in\n"
        "scripts/dead-pub-allow.txt:\n",
        file=sys.stderr,
    )
    for count, name, path in dead:
        print(f"  {name:48} {path.relative_to(ROOT)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
