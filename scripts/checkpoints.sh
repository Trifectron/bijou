#!/usr/bin/env bash
# Downloads nanoDiff checkpoints from the Hugging Face Hub into paths.base_checkpoints.
# With no arguments, downloads the one named by backend.checkpoint. Present files are skipped.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
hub_org=Sebasdi

read -r dir configured < <(uv run --no-sync python -c \
  "from bijou.core.config import load; c = load(); print(c.paths.base_checkpoints, c.backend.checkpoint)")

names=("$@")
[ ${#names[@]} -gt 0 ] || names=("${configured:-}")
if [ -z "${names[0]}" ]; then
  echo "backend.checkpoint is empty; name one: just checkpoints nanodiff-150m-sft-alpaca"
  exit 1
fi

for name in "${names[@]}"; do
  if [ -f "$dir/$name.pt" ]; then
    echo "ok      $dir/$name.pt"
    continue
  fi
  uvx --from huggingface_hub hf download "$hub_org/$name" "$name.pt" --local-dir "$dir"
done
