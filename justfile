# Bijou tasks. Running just with no arguments lists them.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list --unsorted

# ---------- first run ----------

# Check required tools, the submodule, dependencies and checkpoints
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
    uv sync --locked

# Install everything, with CUDA torch
setup-train:
    uv sync --locked --extra train --extra cuda

# Install everything, with CPU torch. What CI uses; the CUDA wheels are gigabytes and CI has no device.
setup-train-cpu:
    uv sync --locked --extra train --extra cpu

# Re-resolve uv.lock after changing dependencies in pyproject.toml
lock:
    uv lock

# Download base checkpoints from the Hugging Face Hub; with no names, the configured one
checkpoints *NAMES:
    ./scripts/checkpoints.sh {{NAMES}}

# Everything a fresh clone needs
bootstrap: env hooks vendor setup
    @echo "ready: 'just check' for the gate; 'just setup-train' and 'just checkpoints' on a GPU box"

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
    uv run pytest -q -m gpu -rs

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

# ---------- deploy ----------

# Build the training image. TORCH=cpu builds one that runs without a GPU
image TORCH="cuda":
    docker build -f deploy/Dockerfile --build-arg TORCH={{TORCH}} \
        --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t bijou-training:{{TORCH}} .

# ---------- housekeeping ----------

clean:
    rm -rf .venv .mypy_cache .pytest_cache .ruff_cache *.egg-info
    find . -name __pycache__ -type d -prune -exec rm -rf {} +
