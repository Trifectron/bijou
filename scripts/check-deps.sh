#!/usr/bin/env bash
# The dependency rule: core <- {adapters, routing, skills, backends} <- runtime <- experiments.
# Contracts live in pyproject.toml [tool.importlinter]. Adding an edge means editing them and
# docs/ARCHITECTURE.md in the same change.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv run lint-imports --config pyproject.toml
