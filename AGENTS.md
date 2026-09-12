# Bijou — agent guide

LoRA skills on a masked diffusion language model, and an agent that plans with an LLM and equips
those skills per step. Read `docs/ROADMAP.md` for what is being built and tested in what order,
and `docs/ARCHITECTURE.md` for the packages, their boundaries and invariants. Do not contradict
them; propose an edit to the doc instead.

## Commands

`just` is the entrypoint (`just` lists recipes). Install: https://just.systems.

```
just bootstrap        # first run: .env, hooks, submodule, dependencies
just doctor           # what is installed, downloaded and running
just setup [cpu|cuda] # every app; cpu or cuda adds the model stack
just checkpoints      # the configured base checkpoint, from the Hugging Face Hub
just check            # format, lint, layering, types, tests — the gate; CI and the hook run it
just test [model|gpu] # tests; model insists on torch, gpu runs the GPU tests
just fmt              # format in place
just lock             # re-resolve uv.lock after changing any pyproject.toml

just chat             # talk to the agent; each message continues the last, /new starts over
just agent "..."      # one request: plan, equip skills, act, answer
just skills [cmd]     # list (default), sample, grade, train, propose, specs, new, collect
just sessions [query] # sessions, newest first or matching; --show <id>
just matrix [--train] # the composition matrix; --train trains everything first
just runs [id]        # run records; an id prints one
just config [table]   # the resolved configuration
just evals [cmd]      # golden cases through engine run --json, gated on the baseline
just console          # the developer console (TUI); alias: just cli
just up <services>    # compose: phoenix, prometheus, grafana, chat (llama-server), gpu-exporter
just down | logs      # stop the compose services, follow their logs
```

A change is not done until `just check` passes.

## Layout

One repo, a uv workspace of three apps. Language is never a folder and neither is model size.

```
apps/engine     everything that runs: the diffusion model, its skill bank, the agent, collection,
                the research matrix and the engine command
apps/evals      golden cases through the engine command, suites, baseline gate
apps/cli        the developer console
third_party/    the nanoDiff submodule, pristine
docs/           ROADMAP.md, ARCHITECTURE.md
deploy/         the engine image
data/           collected skills and proposals (ignored)
runs/           immutable run records (ignored)
```

Each app directory is its own package (`apps/engine` is `engine`), with its own
`pyproject.toml` and `tests/`. A new subpackage is added to that app's `[tool.setuptools]`
packages.

## The dependency rule

Apps never import each other. evals runs `engine run --json` as a subprocess; the console runs
`just` recipes and drives `engine chat --jsonl`. Inside the engine a package imports one below it,
never a sibling:

```
commands      the engine command
wiring        builds everything
agent | clients | tools | memory | collect
runtime       train, evaluate, the matrix, the skill bank
telemetry     metrics and spans, built from trace events
{adapters, routing, skills} and backends
core          config, types, protocols, doubles, runs
```

`scripts/check-deps.sh` runs the contracts in the root `pyproject.toml` under
`[tool.importlinter]`. Adding an edge means editing the contracts and `docs/ARCHITECTURE.md` in the
same change.

`engine.skills` imports no torch and no network client. Only `engine.backends` imports nanoDiff.
The agent side (`agent`, `tools`, `memory`) never imports the model stack; it reaches the
diffusion model only through `SkillRuntime`. Only `clients`, `tools` and `collect` open
connections.

## The seams

All in `engine/core/protocols.py`. The diffusion model: `AdapterSite`, `ActivationPolicy`,
`Grader`, `Skill`, `Backend`. The agent: `ChatModel`, `SkillRuntime`, `Tool`, `Policy`,
`TraceSink`, `SessionStore`, each with a double in `engine/core/doubles.py`. A new replaceable
dependency gets a protocol and a double in the same change.

`SkillRuntime` has one implementation in `engine/clients`: `LocalBank`, the skill bank in the
agent's own process, loaded on first use.

## Config

One `bijou.toml`, committed, shared by every app; each reads only its own tables (the engine the
unprefixed model tables plus `[bank]`, `[collect]` and `[agent.*]`, evals `[evals]`, the console
`[console]`). `BIJOU_<TABLE>__<KEY>` from the environment wins. `BIJOU_CONFIG_FILE` points
elsewhere.

**Every tunable value goes in `bijou.toml`.** `.env` holds only secrets and per-machine URLs; a new
one goes in `.env.example` too. A new setting goes in `bijou.toml` at its default in the same
change. Reject a bad combination at load rather than clamping it at use.

## Rules

- Lints are the law (`[tool.ruff.lint]`): no bare `print` outside commands, annotations on every
  function, imports sorted. `mypy --strict` covers the engine's `core`, `routing`, `skills`,
  `agent`, `tools`, `memory`, the chat client, and evals. Fix at the source
  rather than adding a `noqa`.
- No global mutable state except `AdapterState`. Per-run state goes in `RequestContext`.
- Every train, evaluate and collect run writes a `RunRecord`. A number that is not in a run
  record or an eval report does not go in a table, a doc, or a message.
- `train.seed` and `eval.seed` are never equal. Dataset skill splits are files, fixed at collection.
- `uv.lock` is committed and every install is `--locked`.
- A configured base checkpoint that is missing is an error, never random weights.
- A grader never loads a model, touches a GPU, or calls a network.
- Every write-class tool goes through `Policy`; anything consequential is confirmed against its
  exact payload. A skill spec is approved by a person, never by code.
- Agent answers are model output and never become skill data except through a reviewed spec.
- Errors are `EngineError` subclasses from `engine/core/types/errors.py` (`EvalsError` in evals).
  No bare `Exception` raised, no `assert` for control flow outside tests.
- Never edit `third_party/`. Upstream behaviour we depend on gets a parity test.
- Comment style: see the `comment-style` skill. Plain ASCII, state what the code does, no
  rationale — the why belongs in the commit message.
- Commit messages: imperative subject at most 72 chars, body explains why.

## Skills

`.claude/skills/README.md` lists them. Use `test-driven-development` for features and fixes,
`systematic-debugging` for bugs, `verification-before-completion` before saying anything is done,
`comment-style` when writing comments, and `code-quality` for cleanup passes.

## Out of scope

See "Out of scope" in `docs/ROADMAP.md`.
