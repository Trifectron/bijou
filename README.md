<h1 align="center">Bijou</h1>

<p align="center">
  <em>LoRA skills on a masked diffusion language model, and an agent that equips them.</em>
</p>

<p align="center">
  <a href="https://github.com/Trifectron/bijou/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Trifectron/bijou/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg">
</p>

---

A frozen masked diffusion LM carries a bank of **skills**: named LoRA deltas, each trained on one
narrow task, switchable per request and per point along the denoising trajectory. An **agent**
runs in the same engine. An LLM plans a request into steps and picks which skills each step
equips; subagents act through tools and MCP servers, and use the equipped skills
through the diffusion model. Every run is an indexed session, and recurring work no skill covers
is proposed as a new skill, collected with a teacher model, reviewed, and trained into the bank.

```
User -> planner (LLM) -> subagent per step
                            |-- selector (LLM): which skills to equip
                            |-- run_skill -> skill bank: diffusion base + equipped LoRAs
                            |-- tools and MCP servers (confirmation before anything consequential)
        sessions (indexed) -> pattern miner -> skill spec -> person approves
                           -> collect (teacher LLM) -> skills train -> the bank
```

<img width="2372" height="1340" alt="image" src="https://github.com/user-attachments/assets/4472b03d-2fb4-4a05-8eb8-26c81a70d5bb" />


Alongside the product sits a research track with kill criteria: do adapters beat a tuned prompt,
do they compose, and does routing them by denoising phase beat leaving them on.
`docs/ROADMAP.md` has both tracks; `docs/ARCHITECTURE.md` explains the split between the LLM and
the diffusion model.

## Quick start

```bash
git clone --recurse-submodules https://github.com/Trifectron/bijou
cd bijou
just bootstrap        # .env, git hooks, submodule, every app without torch
just check            # the gate: format, lint, layering, types, tests
```

Then, with a chat model on an OpenAI-compatible endpoint (llama-server by default, set in `.env`):

```bash
just setup cuda && just checkpoints      # torch and the base model, on a GPU box
just skills train json_extract           # one skill in the bank
just agent "Turn this into JSON: Ana has worked as an engineer in Tempe for 7 years."
just chat                                # talk to it; each message continues the last
```

`deploy/README.md` covers the services and the image.

## Apps

One uv workspace, three apps that never import each other.

| App | What it is |
|---|---|
| `apps/engine` | everything that runs: adapters, phase routing, skills, training, the skill bank, the agent (planner, selector, subagents, policy, tools, MCP, sessions, pattern miner), collection, the research matrix and the `engine` command |
| `apps/evals` | golden cases through the `engine` command, scored by suite, gated on a baseline |
| `apps/cli` | a terminal UI over every `just` recipe, with GPU and service status |

`docs/ARCHITECTURE.md` has the layers inside the engine, the request lifecycle, the loop, the risk
classes, and the invariants.

## The engine command

```
engine run "..." [--resume ID]     plan, equip, act, answer; asks before consequential actions
engine confirm SESSION TOKEN       approve (or --deny) a waiting action
engine chat [--jsonl]              talk to the agent, turn after turn; --jsonl is for the console
engine sessions [QUERY] [--show]   the session index
engine skills list|sample|grade|train|propose|specs|new|collect
engine matrix [--train]            the composition matrix
engine runs [ID]                   run records
engine config [TABLE]              the resolved configuration

evals run|compare|baseline|cases   agent evals; run compares against the baseline
```

## Console

```bash
just console          # or: just cli
```

Units on the left, the selected unit's output in the middle, the chat with the agent on the right.
The agent starts with the console, and the logs of every compose service that is up are followed
from the start. `i` types to the agent and `enter` sends; while it works its plan, skill picks,
model and tool calls stream into the middle pane, with a metrics pane under them counting runs,
model calls, tokens, tools, policy decisions and the skill bank (`m` hides it). `a` and `d` approve or deny an action it is
holding, `R` starts a new conversation. `j`/`k` move and `enter` starts or stops the selected
unit — on a service row that is the compose service itself, whose logs then follow while it runs.
`h`/`l` change pane, `:` runs any recipe, `/` searches, `?` for help, and `g` or `t` hands the
whole terminal to nvtop or htop until you quit it. The status bar shows the agent, the base
checkpoint, trained skills, the last run and the git SHA. Every line a unit prints is mirrored to
`.bijou/logs/<unit>.log`.

## Observability

```bash
just up phoenix prometheus grafana   # traces :6006, metrics :9090, dashboards :3000 (admin/admin)
just up chat                         # llama-server :8000, unless it already runs on the host
```

Set `BIJOU_TELEMETRY__OTLP_ENDPOINT=http://127.0.0.1:6006/v1/traces` in `.env` and every run
shows in Phoenix as a span tree: the run, each step with the skills it equipped, every model call
with its prompt and reply, every tool call. `engine chat` serves Prometheus metrics at
`telemetry.metrics_port` (9464) for as long as it runs; Grafana's `Bijou` dashboard covers the
agent, the skill bank, llama-server and the GPU.

## Safety

Tools carry a risk class. Reads run; anything that submits, clicks, posts or deletes waits for a
confirmation bound to its exact payload, single use and short lived. Unrecognised MCP tools count
as consequential. Authenticated reads are off. A skill spec is approved by a person, never by code.

## Reproducibility

Every train, evaluate and collect run writes an immutable record under `runs/` with the resolved
config, the git SHA, the environment, input digests and scores. Every agent run writes a JSONL
trace under `.bijou/traces/`. One committed `bijou.toml` holds every setting; `BIJOU_*` environment
variables override it; contradictory settings are rejected at load.

## Naming

These are **skills** or **adapters**. The v0 design calls the bank "SLoRA skills"; in the code
they are never called that, because [S-LoRA](https://arxiv.org/abs/2311.03285) is an existing
serving system.

## Substrate

[nanoDiff](https://github.com/BY571/nanoDiff), pinned as a submodule: LLaDA-style masked diffusion
at 50M, 150M and 350M, chosen for iteration speed. A larger diffusion
backend is a sibling module in `apps/engine/backends`, on the agent roadmap.

Contributing: read [AGENTS.md](AGENTS.md).

## License

MIT.
