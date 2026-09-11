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

# Install every app without the model stack (graders, schedules, config, the agent, evals)
setup:
    uv sync --locked --all-packages

# Install every app, with CUDA torch for the engine
setup-train:
    uv sync --locked --all-packages --extra train --extra cuda

# Install every app, with CPU torch. What CI uses; the CUDA wheels are gigabytes and CI has no device.
setup-train-cpu:
    uv sync --locked --all-packages --extra train --extra cpu

# Re-resolve uv.lock after changing dependencies in any pyproject.toml
lock:
    uv lock

# Download base checkpoints from the Hugging Face Hub; with no names, the configured one
checkpoints *NAMES:
    ./scripts/checkpoints.sh {{NAMES}}

# Everything a fresh clone needs
bootstrap: env hooks vendor setup
    @echo "ready: 'just check' for the gate; 'just setup-train' and 'just checkpoints' on a GPU box"

# ---------- console ----------

# Developer console: run recipes, stream their logs, watch the GPU, services and artifacts
console:
    uv run console

alias cli := console

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

# The dependency rule from pyproject.toml [tool.importlinter], between apps and inside each
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

# ---------- services ----------

# The engine over HTTP on agent.http: the agent, with the skill bank in process by default
serve:
    uv run engine serve

# The skill bank alone over HTTP on serve.port, for an agent elsewhere (agent.skills.mode = http)
serve-skills:
    uv run engine serve-skills

# Playwright MCP over Streamable HTTP, the browser the agent drives. Needs node
browser PORT="8931":
    npx -y @playwright/mcp@latest --port {{PORT}}

# ---------- the agent ----------

# Run one request: plan, equip skills, act, answer
agent +REQUEST:
    uv run engine run "{{REQUEST}}"

# The engine command: run, confirm, serve, skills, sessions, patterns, skill, collect, runs
engine *ARGS:
    uv run engine {{ARGS}}

# Sessions: list, show <id>, search <text>
sessions *ARGS:
    uv run engine sessions {{ARGS}}

# Recurring work no skill covers; --write turns each into a skill spec for review
patterns *ARGS:
    uv run engine patterns {{ARGS}}

# Skill specs: list, new, run <spec>. Collection needs an approved spec and the teacher model
collect *ARGS:
    uv run engine collect {{ARGS}}

# ---------- experiments ----------

# Resolved configuration
config:
    uv run engine config

# Skills: list, sample, train
skill *ARGS:
    uv run engine skill {{ARGS}}

# Score the composition matrix
evaluate:
    uv run engine evaluate

# Run records
runs:
    uv run engine runs list

# Train an adapter and a full fine-tune for every skill, then score the matrix
matrix:
    #!/usr/bin/env bash
    set -euo pipefail
    for s in $(uv run engine skill names); do
        uv run engine skill train "$s"
        uv run engine skill train "$s" --full-finetune
    done
    uv run engine evaluate

# ---------- evals ----------

# Evals: run, compare, baseline, cases. run needs just serve
evals *ARGS:
    uv run evals {{ARGS}}

# ---------- deploy ----------

# Build the training image. TORCH=cpu builds one that runs without a GPU
image TORCH="cuda":
    docker build -f deploy/Dockerfile --build-arg TORCH={{TORCH}} \
        --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t bijou-training:{{TORCH}} .

# ---------- housekeeping ----------

clean:
    rm -rf .venv .mypy_cache .pytest_cache .ruff_cache .import_linter_cache
    find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
    find . -name __pycache__ -type d -prune -exec rm -rf {} +
