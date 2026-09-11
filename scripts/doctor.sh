#!/usr/bin/env bash
# Checks that every tool the repo needs is installed and prints versions.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
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
echo "python:"
if uv sync --locked --all-packages --inexact --check >/dev/null 2>&1; then
  echo "  ok      dependencies"
else
  echo "  MISSING dependencies  run: just setup"; ok=1
fi
torch=$(uv run --no-sync python -c "import torch; print(torch.__version__)" 2>/dev/null)
[ -n "$torch" ] && echo "  ok      torch $torch" || echo "  absent  torch       run: just setup cuda (or cpu)"
echo "checkpoints:"
checkpoint=$(uv run --no-sync python -c \
  "from engine.core.config import load; c = load(); print(c.paths.base_checkpoints / (c.backend.checkpoint + '.pt') if c.backend.checkpoint else '')" 2>/dev/null)
if [ -z "$checkpoint" ]; then
  echo "  none configured (random weights)"
elif [ -f "$checkpoint" ]; then
  echo "  ok      $checkpoint"
else
  echo "  absent  $checkpoint  run: just checkpoints"
fi
echo "services:"
probe() {
  if curl -fsS -m 2 "$2" >/dev/null 2>&1; then
    printf '  ok      %-8s %s\n' "$1" "$2"
  else
    printf '  down    %-8s %s  %s\n' "$1" "$2" "$3"
  fi
}
llm=$(uv run --no-sync python -c \
  "from engine.core.config import load; print(load().agent.llm.base_url)" 2>/dev/null)
[ -n "$llm" ] && probe "chat" "${llm%/}/models" "start llama-server, or set BIJOU_AGENT__LLM__BASE_URL"
metrics=$(uv run --no-sync python -c \
  "from engine.core.config import load; t = load().telemetry; \
print(f'http://{t.metrics_host}:{t.metrics_port}/metrics' if t.metrics_port else '')" 2>/dev/null)
[ -n "$metrics" ] && probe "agent" "$metrics" "run: just chat, or just console"
probe "phoenix" "http://127.0.0.1:6006/" "run: just up observe"
probe "prom" "http://127.0.0.1:9090/-/ready" "run: just up observe"
probe "grafana" "http://127.0.0.1:3000/api/health" "run: just up observe"
echo "gpu:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  /'
else
  echo "  none (CPU tests still run; just test-gpu needs one)"
fi
exit $ok
