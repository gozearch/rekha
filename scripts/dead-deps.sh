#!/usr/bin/env bash
# Fail when a workspace crate declares a dependency it never uses.
set -euo pipefail

# Version, not "latest": 0.10.1 uses `Path::is_empty`, which is unstable on the
# 1.97 toolchain this repo pins, so an unpinned install does not compile.
if ! command -v cargo-machete >/dev/null 2>&1; then
  echo "cargo-machete is not installed. Install the version CI uses:" >&2
  echo "  cargo install cargo-machete --version 0.9.2 --locked" >&2
  exit 1
fi

cargo metadata --no-deps --format-version 1 \
  | python3 -c 'import json,sys; [print(p["manifest_path"].rsplit("/", 1)[0]) for p in json.load(sys.stdin)["packages"]]' \
  | xargs cargo machete
