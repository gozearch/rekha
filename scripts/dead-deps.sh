#!/usr/bin/env bash
# Fail when a workspace crate declares a dependency it never uses.
set -euo pipefail

cargo metadata --no-deps --format-version 1 \
  | python3 -c 'import json,sys; [print(p["manifest_path"].rsplit("/", 1)[0]) for p in json.load(sys.stdin)["packages"]]' \
  | xargs cargo machete
