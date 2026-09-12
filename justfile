# Bijou tasks. Running just with no arguments lists them.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list --unsorted

# ---------- first run ----------

# Check required tools, the submodule, dependencies, checkpoints and services
doctor:
    ./scripts/doctor.sh

# Everything a fresh clone needs: .env, git hooks, the submodule, dependencies
bootstrap: env hooks vendor setup
    @echo "ready: 'just check' for the gate; 'just setup cuda' and 'just checkpoints' on a GPU box"

# Install every app; TORCH=cpu or cuda adds the model stack (CI uses cpu)
setup TORCH="":
    @case "{{TORCH}}" in ""|cpu|cuda) ;; *) echo "TORCH is cpu or cuda"; exit 1;; esac
    uv sync --locked --all-packages {{ if TORCH == "" { "" } else { "--extra train --extra " + TORCH } }}

# Re-resolve uv.lock after changing dependencies in any pyproject.toml
lock:
    uv lock

# Download base checkpoints from the Hugging Face Hub; with no names, the configured one
checkpoints *NAMES:
    ./scripts/checkpoints.sh {{NAMES}}

[private]
env:
    @[ -f .env ] && echo ".env exists" || { cp .env.example .env && echo "created .env"; }

[private]
hooks:
    git config core.hooksPath .githooks

[private]
vendor:
    git submodule update --init --recursive

# ---------- the gate ----------

# Format check, lint, layering, types, tests. CI and the hook run this.
check: fmt-check lint deps types test
    @echo "ok"

# Format in place
fmt:
    uvx ruff format .
    uvx ruff check --fix .

# Tests without a GPU; MODE=model insists on torch so nothing skips, MODE=gpu runs the GPU tests
test MODE="":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{MODE}}" in
      "") uv run pytest -q -m "not gpu" ;;
      model) uv run python -c "import torch; print('torch', torch.__version__)"
             uv run pytest -q -m "not gpu" --no-header -rs ;;
      gpu) uv run pytest -q -m gpu -rs ;;
      *) echo "MODE is model or gpu"; exit 1 ;;
    esac

[private]
fmt-check:
    uvx ruff format --check .

[private]
lint:
    uvx ruff check .

[private]
deps:
    ./scripts/check-deps.sh

[private]
types:
    uv run mypy

# ---------- the engine ----------

# Talk to the agent; each message continues the conversation, /new starts over
chat *ARGS:
    uv run engine chat {{ARGS}}

# One request: plan, equip skills, act, answer
agent +REQUEST:
    uv run engine run "{{REQUEST}}"

# The skill bank: list, sample, grade, train, propose, specs, new, collect
skills *ARGS="list":
    uv run engine skills {{ARGS}}

# Sessions, newest first, or matching a query; --show <id> for one in full
sessions *ARGS:
    uv run engine sessions {{ARGS}}

# Score the composition matrix; --train trains every adapter and full fine-tune first
matrix *ARGS:
    uv run engine matrix {{ARGS}}

# Run records, newest first; a run id prints one in full
runs *ARGS:
    uv run engine runs {{ARGS}}

# The resolved configuration, or one table of it
config *ARGS:
    uv run engine config {{ARGS}}

# Any engine command
engine *ARGS:
    uv run engine {{ARGS}}

# ---------- the services ----------

# Start compose services by name: chat (llama-server), phoenix, prometheus, grafana, gpu-exporter
up +SERVICES:
    docker compose -f deploy/compose.yml --profile '*' up -d {{SERVICES}}

# Stop compose services by name, or every one of them
stop *SERVICES:
    docker compose -f deploy/compose.yml --profile '*' stop {{SERVICES}}

# Stop and remove every compose service
down:
    docker compose -f deploy/compose.yml --profile '*' down

# Follow the logs of compose services, all of them by default
logs *SERVICES:
    docker compose -f deploy/compose.yml --profile '*' logs -f --tail 200 {{SERVICES}}

# ---------- evals ----------

# Golden cases, each through engine run --json, gated on the baseline; also compare, baseline, cases
evals *ARGS="run":
    uv run evals {{ARGS}}

# ---------- console and deploy ----------

# Developer console: run recipes, stream their logs, watch the GPU and the engine
console:
    uv run console

alias cli := console

# The GPU in nvtop, full screen
nvtop:
    nvtop

# Processes and CPU in htop, full screen
htop:
    htop

# Build the engine image; TORCH=cpu builds one that runs without a GPU
image TORCH="cuda":
    docker build -f deploy/Dockerfile --build-arg TORCH={{TORCH}} \
        --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t bijou-training:{{TORCH}} .

clean:
    rm -rf .venv .mypy_cache .pytest_cache .ruff_cache .import_linter_cache
    find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
    find . -name __pycache__ -type d -prune -exec rm -rf {} +
