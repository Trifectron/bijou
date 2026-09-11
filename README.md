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
equips; subagents act through tools, MCP servers and a browser, and use the equipped skills
through the diffusion model. Every run is an indexed session, and recurring work no skill covers
is proposed as a new skill, collected with a teacher model, reviewed, and trained into the bank.

```
User -> planner (LLM) -> subagent per step
                            |-- selector (LLM): which skills to equip
                            |-- run_skill -> skill bank: diffusion base + equipped LoRAs
                            |-- tools, MCP, browser (confirmation before anything consequential)
        sessions (indexed) -> pattern miner -> skill spec -> person approves
                           -> collect (teacher LLM) -> skill train -> the bank
```

Alongside the product sits a research track with kill criteria: do adapters beat a tuned prompt,
do they compose, and does routing them by denoising phase beat leaving them on.
`docs/ROADMAP.md` has both tracks; `docs/decisions/0002-*` explains the split between the LLM and
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
just setup-train && just checkpoints     # torch and the base model, on a GPU box
just skill train json_extract            # one skill in the bank
just agent "Turn this into JSON: Ana has worked as an engineer in Tempe for 7 years."
just serve                               # the engine over HTTP, for clients and evals
```

`deploy/README.md` covers the services and the image.

## Apps

One uv workspace, three apps that never import each other.

| App | What it is |
|---|---|
| `apps/engine` | everything that runs: adapters, phase routing, skills, training, the skill bank, the agent (planner, selector, subagents, policy, tools, MCP, sessions, pattern miner), collection, the research matrix, the HTTP surface and the `engine` command |
| `apps/evals` | golden cases against a serving engine, scored by suite, gated on a baseline |
| `apps/cli` | a terminal UI over every `just` recipe, with GPU and service status |

`docs/ARCHITECTURE.md` has the layers inside the engine, the request lifecycle, the loop, the risk
classes, and the invariants.

## The engine command

```
engine run "..." [--resume ID]     plan, equip, act, answer; asks before consequential actions
engine confirm SESSION TOKEN       approve (or --deny) a waiting action
engine serve                       the agent over HTTP: /run /confirm /sessions /skills /patterns
engine serve-skills                the skill bank alone over HTTP, for an agent elsewhere
engine sessions list|search|show   the session index
engine skills                      what the skill bank can equip
engine patterns [--write]          recurring uncovered work, as skill specs for review
engine collect list|new|run        skill specs; an approved spec becomes a dataset skill
engine skill list|sample|grade|train
engine evaluate                    the composition matrix
engine runs list|show              run records
engine config [table]              the resolved configuration

evals run|compare|baseline|cases   agent evals
```

## Console

```bash
just console          # or: just cli
```

Units on the left, the selected unit's output on the right; `j`/`k` move, `enter` starts or stops,
`:` runs any recipe, `/` searches, `?` for help. The status bar shows GPU memory and its holders,
the base checkpoint, trained skills, whether the engine answers, the last run, and the git SHA.
Every line is mirrored to `.bijou/logs/<unit>.log`.

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
at 50M, 150M and 350M, chosen for iteration speed (`docs/decisions/0001-*`). A larger diffusion
backend is a sibling module in `apps/engine/backends`, on the agent roadmap.

Contributing: read [AGENTS.md](AGENTS.md).

## License

MIT.
