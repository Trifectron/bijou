# Bijou — agent guide

LoRA skills on a masked diffusion language model, and an agent that plans with an LLM and equips
those skills per step. Read `docs/ROADMAP.md` for what is being built and tested in what order,
`docs/ARCHITECTURE.md` for the packages, their boundaries and invariants, and `docs/decisions/`
for why. Do not contradict them; propose an edit to the doc instead.

## Commands

`just` is the entrypoint (`just` lists recipes). Install: https://just.systems.

```
just doctor | env | hooks | vendor | bootstrap   # first run
just check            # fmt-check, lint, layering, types, tests — the gate; CI and the hook run it
just fmt              # format in place
just setup            # every app, no torch
just setup-train      # plus CUDA torch for the engine; setup-train-cpu for CPU torch
just checkpoints      # the configured base checkpoint, from the Hugging Face Hub
just lock             # re-resolve uv.lock after changing any pyproject.toml
just console          # the developer console (TUI); alias: just cli

just serve            # the engine over HTTP: the agent, with the skill bank in process
just serve-skills     # the skill bank alone, for an agent elsewhere
just browser          # Playwright MCP, the browser the agent drives
just agent "..."      # one request: plan, equip skills, act, answer
just engine ...       # the engine command: run, confirm, skills, sessions, patterns, skill, ...
just patterns --write # recurring uncovered work as skill specs for review
just collect ...      # list, new, run <spec>: approved spec -> dataset skill
just evals ...        # run, compare, baseline, cases, against a serving engine

just skill list | sample <name> | train <name>
just evaluate         # score the composition matrix
just runs             # every run record
just test-gpu         # the tests needing a GPU and a base checkpoint
```

A change is not done until `just check` passes.

## Layout

One repo, a uv workspace of three apps. Language is never a folder and neither is model size.

```
apps/engine     everything that runs: the diffusion model, its skill bank, the agent, collection,
                the research matrix, the HTTP surface and the engine command
apps/evals      golden cases against a serving engine, suites, baseline gate
apps/cli        the developer console
third_party/    the nanoDiff submodule, pristine
docs/           ROADMAP.md, ARCHITECTURE.md, decisions/
deploy/         the engine image
data/           collected skills and proposals (ignored)
runs/           immutable run records (ignored)
```

Each app directory is its own package (`apps/engine` is `engine`), with its own
`pyproject.toml` and `tests/`. A new subpackage is added to that app's `[tool.setuptools]`
packages.

## The dependency rule

Apps never import each other. evals reaches the engine over HTTP (`/run`); the console runs
`just` recipes. Inside the engine a package imports one below it, never a sibling:

```
commands
routes | experiments
wiring
agent | model | tools | stores | patterns | collect
runtime
{adapters, routing, skills} and backends
core
```

`scripts/check-deps.sh` runs the contracts in the root `pyproject.toml` under
`[tool.importlinter]`. Adding an edge means editing the contracts and `docs/ARCHITECTURE.md` in the
same change.

`engine.skills` imports no torch and no network client. Only `engine.backends` imports nanoDiff.
The agent (`agent`, `tools`, `stores`, `patterns`) never imports the model stack; it reaches the
diffusion model only through `SkillRuntime`. Only `model`, `tools` and `collect` open connections.

## The seams

All in `engine/core/protocols.py`. The diffusion model: `AdapterSite`, `ActivationPolicy`,
`Grader`, `Skill`, `Backend`. The agent: `ChatModel`, `SkillRuntime`, `Tool`, `Policy`,
`TraceSink`, `SessionStore`, each with a double in `engine/core/doubles.py`. A new replaceable
dependency gets a protocol and a double in the same change.

`SkillRuntime` has two implementations: `LocalSkillRuntime`, the bank in this process, and
`HttpSkillRuntime`, a bank served by `engine serve-skills`. `agent.skills.mode` picks one.

## Config

One `bijou.toml`, committed, shared by every app; each reads only its own tables (the engine the
unprefixed model tables plus `[serve]`, `[collect]` and `[agent.*]`, evals `[evals]`, the console
`[console]`). `BIJOU_<TABLE>__<KEY>` from the environment wins. `BIJOU_CONFIG_FILE` points
elsewhere.

**Every tunable value goes in `bijou.toml`.** `.env` holds only secrets and per-machine URLs; a new
one goes in `.env.example` too. A new setting goes in `bijou.toml` at its default in the same
change. Reject a bad combination at load rather than clamping it at use.

## Rules

- Lints are the law (`[tool.ruff.lint]`): no bare `print` outside commands, annotations on every
  function, imports sorted. `mypy --strict` covers the engine's `core`, `routing`, `skills`,
  `agent`, `tools`, `stores`, `patterns`, the chat and skill clients, and evals. Fix at the
  source rather than adding a `noqa`.
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
  rationale — the why belongs in the commit message and in `docs/decisions/`.
- Commit messages: imperative subject at most 72 chars, body explains why.

## Skills

`.claude/skills/README.md` lists them. Use `test-driven-development` for features and fixes,
`systematic-debugging` for bugs, `verification-before-completion` before saying anything is done,
`comment-style` when writing comments, `adr` when a decision needs recording, `code-quality` for
cleanup passes.

## Out of scope

See "Out of scope" in `docs/ROADMAP.md`.
