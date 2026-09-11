# Deploy

Two kinds of thing run. One-shot jobs: train an adapter, collect a skill, score the matrix, exit.
And services: the engine (`just serve`), which is the agent with the skill bank in its own process,
and a chat model behind an OpenAI-compatible API (llama-server). evals and the console run from
source anywhere that can reach the engine.

## Services, locally

```bash
just up model                 # llama-server on :8000 in docker; or run it on the host, see inference/
just up observe               # Phoenix :6006, Prometheus :9090, Grafana :3000 (admin/admin)
just serve                    # the engine on agent.http.port, the skill bank in process
just doctor                   # the services section should read ok
just agent "What day is it?"  # or just evals
just down                     # stop every compose service
```

`--jinja` makes llama-server apply the model's chat template, which tool calls need; the compose
service passes it, with `--metrics` for Prometheus. Set
`BIJOU_TELEMETRY__OTLP_ENDPOINT=http://127.0.0.1:6006/v1/traces` in `.env` to send spans to Phoenix.
`just up gpu` adds the GPU exporter behind the inference dashboard.

| Profile | Services | Ports |
|---|---|---|
| `model` | `chat` (llama-server, CUDA) | 8000 |
| `observe` | `phoenix`, `prometheus`, `grafana` | 6006, 4317, 9090, 3000 |
| `gpu` | `gpu-exporter` | 9835 |
| `engine` | `engine`, the image below, pointed at `chat` and `phoenix` | 8200 |

Prometheus and Grafana run on the host network, bound to 127.0.0.1. Prometheus scrapes the engine
(:8200), the bank (:8100), llama-server (:8000) and the GPU exporter (:9835) on the host's
loopback, which also covers those services when they run in compose with their ports published.
The compose project is named `bijou`, so its containers and volumes never collide with another
stack whose compose file also sits in a `deploy/` directory.

## The bank on a GPU box, the agent elsewhere

Set `agent.skills.mode = "http"` and `BIJOU_AGENT__SKILLS__URL` where the agent runs, and on the
GPU box:

```bash
just serve --bank             # the skill bank alone on serve.port
```

## On a GPU box, from source

```bash
git clone --recurse-submodules https://github.com/Trifectron/bijou && cd bijou
just bootstrap        # .env, hooks, submodule, dependencies without torch
just setup cuda       # CUDA torch, pinned by uv.lock
just checkpoints      # the base checkpoint named by backend.checkpoint in bijou.toml
just doctor           # every line should read ok
just matrix --train   # train every configured skill, then score the matrix
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
  ghcr.io/trifectron/bijou/training:main skills train json_extract

docker run --rm --gpus all -p 8200:8200 -e BIJOU_AGENT__HTTP__HOST=0.0.0.0 \
  -e BIJOU_AGENT__LLM__BASE_URL=http://host.docker.internal:8000/v1 \
  -v "$PWD/checkpoints:/app/checkpoints" -v "$PWD/data:/app/data" -v "$PWD/.bijou:/app/.bijou" \
  ghcr.io/trifectron/bijou/training:main serve
```

`serve --bank` in place of `serve`, with `-p 8100:8100 -e BIJOU_SERVE__HOST=0.0.0.0`, runs the
bank alone.

Override any setting with `-e BIJOU_<TABLE>__<KEY>=value`. Run records written from the image
carry the commit it was built from, through `BIJOU_GIT_SHA`.

`just image` builds the same image locally; `just image cpu` builds one with CPU torch, which is
how to check the Dockerfile on a machine without a GPU.
