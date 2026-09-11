#!/usr/bin/env bash
# The vendored nanoDiff is reimplemented in bijou/backends, so a moved pin can change
# behaviour silently. A commit that moves it must also touch a parity test.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pinned=$(git ls-tree HEAD third_party/nanoDiff | awk '{print $3}')
echo "nanoDiff pinned at ${pinned:0:8}"

base="${GITHUB_BASE_REF:-}"
if [ -z "$base" ]; then
  echo "no base ref, nothing to compare"
  exit 0
fi
git fetch --quiet --depth=1 origin "$base"
before=$(git ls-tree "origin/$base" third_party/nanoDiff | awk '{print $3}')
[ "$before" = "$pinned" ] && { echo "pin unchanged"; exit 0; }

echo "pin moved ${before:0:8} -> ${pinned:0:8}"
if git diff --name-only "origin/$base"...HEAD | grep -qE '^tests/'; then
  echo "a test changed alongside it"
  exit 0
fi
echo "FAIL the submodule moved with no test change; add or update a parity test"
exit 1
