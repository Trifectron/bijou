# Deploy

Two kinds of thing run. One-shot jobs: train an adapter, collect a skill, score the matrix, exit.
And services: the engine (`just serve`), which is the agent with the skill bank in its own process,
and a chat model behind an OpenAI-compatible API (llama-server). evals and the console run from
source anywhere that can reach the engine.

## Services, locally

```bash
llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M --port 8000 --jinja   # the chat model
just browser                  # optional: Playwright MCP; set BIJOU_AGENT__MCP__PLAYWRIGHT_URL
just serve                    # the engine on agent.http.port, the skill bank in process
just doctor                   # the services section should read ok
just agent "What day is it?"  # or just evals run
```

`--jinja` makes llama-server apply the model's chat template, which tool calls need.

## The bank on a GPU box, the agent elsewhere

Set `agent.skills.mode = "http"` and `BIJOU_AGENT__SKILLS__URL` where the agent runs, and on the
GPU box:

```bash
just serve-skills             # the skill bank alone on serve.port
```

## On a GPU box, from source

```bash
git clone --recurse-submodules https://github.com/Trifectron/bijou && cd bijou
just bootstrap        # .env, hooks, submodule, dependencies without torch
just setup-train      # CUDA torch, pinned by uv.lock
just checkpoints      # the base checkpoint named by backend.checkpoint in bijou.toml
just doctor           # every line should read ok
just matrix           # train every configured skill, then score the matrix
```

`just checkpoints nanodiff-50m-sft-alpaca nanodiff-350m-base` fetches others by name. The six
published checkpoints are listed in `third_party/nanoDiff/README.md`.

## The image

CD builds `ghcr.io/trifectron/bijou/training` on every push to `main` that changes what the image
contains, tagged with the branch, the short SHA, and a version on `v*` tags. The image carries
`apps/engine`, `bijou.toml` and the locked dependencies; its entrypoint is `engine`. Checkpoints,
adapters, data and run records are mounted.

```bash
docker run --rm --gpus all \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/runs:/app/runs" \
  ghcr.io/trifectron/bijou/training:main skill train json_extract

docker run --rm --gpus all -p 8200:8200 -e BIJOU_AGENT__HTTP__HOST=0.0.0.0 \
  -e BIJOU_AGENT__LLM__BASE_URL=http://host.docker.internal:8000/v1 \
  -v "$PWD/checkpoints:/app/checkpoints" -v "$PWD/data:/app/data" -v "$PWD/.bijou:/app/.bijou" \
  ghcr.io/trifectron/bijou/training:main serve
```

`serve-skills` in place of `serve`, with `-p 8100:8100 -e BIJOU_SERVE__HOST=0.0.0.0`, runs the
bank alone.

Override any setting with `-e BIJOU_<TABLE>__<KEY>=value`. Run records written from the image
carry the commit it was built from, through `BIJOU_GIT_SHA`.

`just image` builds the same image locally; `just image cpu` builds one with CPU torch, which is
how to check the Dockerfile on a machine without a GPU.
