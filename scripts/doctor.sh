#!/usr/bin/env bash
# Checks that every tool the repo needs is installed and prints versions.
set -uo pipefail
ok=0
need() {
  if command -v "$2" >/dev/null 2>&1; then
    printf '  ok      %-8s %s\n' "$1" "$($2 --version 2>/dev/null | head -1)"
  else
    printf '  MISSING %-8s install: %s\n' "$1" "$3"; ok=1
  fi
}
echo "tools:"
need just just "https://just.systems"
need uv   uv   "https://docs.astral.sh/uv"
need git  git  "package manager"
echo "repo:"
[ -f third_party/nanoDiff/nanodiff/model.py ] && echo "  ok      submodule" || { echo "  MISSING submodule   run: just vendor"; ok=1; }
[ -f .env ] && echo "  ok      .env" || echo "  MISSING .env        run: just env"
[ "$(git config core.hooksPath)" = ".githooks" ] && echo "  ok      git hooks" || echo "  MISSING git hooks   run: just hooks"
echo "gpu:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  /'
else
  echo "  none (CPU tests still run; just test-gpu needs one)"
fi
exit $ok
