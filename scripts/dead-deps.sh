#!/usr/bin/env bash
# Fail when a workspace crate declares a dependency it never uses.
#
# `cargo machete` scans every Cargo.toml under the tree, including crates that
# are excluded from the workspace. This repository deliberately shelves two of
# those (`rekha-repl`, `rekha-objectstore`), and an unused dependency in a crate
# nothing compiles is not a signal about the shipping binary. Ask cargo for the
# actual members and check only those.
#
# The alternative — an `ignored` list in each shelved manifest — would hide a
# real unused dependency if the crate is ever revived.
set -euo pipefail

cargo metadata --no-deps --format-version 1 \
  | python3 -c 'import json,sys; [print(p["manifest_path"].rsplit("/", 1)[0]) for p in json.load(sys.stdin)["packages"]]' \
  | xargs cargo machete
