# Bijou tasks. Running just with no arguments lists them.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list --unsorted

# ---------- first run ----------

# Check required tools and the submodule
doctor:
    ./scripts/doctor.sh

# Create .env from the example (no-op if it exists). Settings live in bijou.toml
env:
    @[ -f .env ] && echo ".env exists" || { cp .env.example .env && echo "created .env"; }

# Install the pre-commit hook (runs the gate)
hooks:
    git config core.hooksPath .githooks
    @echo "hooks installed: .githooks/pre-commit"

# Fetch the nanoDiff submodule
vendor:
    git submodule update --init --recursive

# Install dependencies without the model stack (graders, schedules, config)
setup:
    uv sync --extra dev

# Install everything including torch
setup-train:
    uv sync --extra dev --extra train

# Install with CPU-only torch. What CI uses; the CUDA wheel is 2 GB and CI has no device.
setup-train-cpu:
    uv sync --extra dev --extra train --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Everything a fresh clone needs
bootstrap: env hooks vendor setup
    @echo "ready: 'just check' for the gate, 'just skill list' to see what exists"

# ---------- the gate ----------

# Format check, lint, layering, types, tests. CI and the hook run this.
check: fmt-check lint deps types test
    @echo "ok"

# Format in place
fmt:
    uvx ruff format .
    uvx ruff check --fix .

fmt-check:
    uvx ruff format --check .

lint:
    uvx ruff check .

# The dependency rule from pyproject.toml [tool.importlinter]
deps:
    ./scripts/check-deps.sh

# The vendored submodule pin moved only alongside a test change
vendor-check:
    ./scripts/check-vendor.sh

types:
    uv run mypy

# Everything not needing a GPU. Tests importing torch skip when it is absent.
test:
    uv run pytest -q -m "not gpu"

# The same tests with the model stack installed, so nothing skips silently.
check-model:
    uv run python -c "import torch; print('torch', torch.__version__)"
    uv run pytest -q -m "not gpu" --no-header -rs

# Tests that need a GPU and a base checkpoint
test-gpu:
    uv run pytest -q -m gpu

# ---------- experiments ----------

# Resolved configuration
config:
    uv run bijou config

# Skills: list, sample, train
skill *ARGS:
    uv run bijou skill {{ARGS}}

# Score the composition matrix
evaluate:
    uv run bijou evaluate

# Run records
runs:
    uv run bijou run list

# Train every configured skill, then score the matrix
matrix:
    #!/usr/bin/env bash
    set -euo pipefail
    for s in $(uv run bijou skill names); do
        uv run bijou skill train "$s"
    done
    uv run bijou evaluate

# ---------- housekeeping ----------

clean:
    rm -rf .venv .mypy_cache .pytest_cache .ruff_cache
    find . -name __pycache__ -type d -prune -exec rm -rf {} +
