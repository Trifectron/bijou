# Bijou — agent guide

Modular capability deltas for masked diffusion language models. Read `docs/ROADMAP.md` for what
we are testing and in what order, and `docs/ARCHITECTURE.md` for the package boundaries and the
dependency rule. Do not contradict either — propose an edit to the doc instead.

This is a research repo with kill criteria, not a product. A change that does not move an
experiment forward, or make one reproducible, probably does not belong.

## Commands

`just` is the entrypoint (`just` lists recipes). Install: https://just.systems.

```
just doctor | env | hooks | vendor | bootstrap   # first run
just check            # fmt-check, lint, layering, types, tests — the gate; CI and the hook run it
just fmt              # format in place
just setup            # deps without torch
just setup-train      # deps with torch
just config           # the resolved configuration
just skill list | sample <name> | train <name>
just evaluate         # score the composition matrix
just runs             # every run record
just matrix           # train every configured skill, then score the matrix
just test-gpu         # the tests needing a GPU and a base checkpoint
```

A change is not done until `just check` passes.

## Layout

One repo, one package. Language is never a folder and neither is model size.

```
bijou/core          types, protocols, config, determinism, run records. Imports nothing else.
bijou/adapters      the AdapterSite seam: LoRA injection, weights on disk
bijou/routing       the ActivationPolicy seam: phase schedules and the router
bijou/skills        one module per skill, each generate(n, seed) and grade(sample, output)
bijou/backends      the only package importing third_party.nanoDiff
bijou/runtime       composes a backend, adapters and a policy into train and evaluate
bijou/experiments   the matrix runner and the bijou CLI
third_party/        the nanoDiff submodule, pristine
configs/            experiment configs that override bijou.toml
runs/               immutable run records
docs/               ROADMAP.md, ARCHITECTURE.md, experiments.md, decisions/
deploy/             the training image
```

## The dependency rule

A package may import one listed below it. A sibling is never imported.

```
experiments -> runtime -> {adapters, routing, skills, backends} -> core
```

`scripts/check-deps.sh` runs the contracts in `pyproject.toml` under `[tool.importlinter]`, in
`just check`, the pre-commit hook, and CI. Adding an edge means editing the contracts and
`docs/ARCHITECTURE.md` in the same change.

`bijou.skills` imports no torch, which is what keeps graders unit-testable in milliseconds.
`bijou.adapters` and `bijou.routing` never import each other; `AdapterState` lives in `core` so a
routing policy is testable with no adapter implementation present.

## The three seams

Everything else is infrastructure for these. Change them deliberately.

`core.protocols.AdapterSite` — how a delta attaches to one frozen module. LoRA is one
implementation. A second one costs nothing outside `bijou/adapters`.

`core.protocols.ActivationPolicy` — what is live at denoising step t. Static application is a
schedule with one phase, so the static and phase-routed conditions run the same code path and the
comparison stays controlled.

`core.protocols.Grader` — `score(sample, output) -> Score`. One signature for every skill, which
is what makes the composition matrix a loop instead of N scripts.

`Backend` is the fourth interface. It exists so a move to a larger diffusion LM is a sibling
module, not a rewrite.

## Config

Two layers, lowest first: `bijou.toml`, which is committed, and `BIJOU_<SECTION>__<KEY>` from the
environment, which wins. `BIJOU_CONFIG_FILE` points elsewhere.

**Every tunable value goes in `bijou.toml`.** `.env` holds only secrets and per-machine paths. A
new setting goes in `bijou.toml` at its default in the same change. Reject a bad combination in
`Config.validate_combinations` at load rather than clamping it at use.

## Rules

- Lints are the law (`[tool.ruff.lint]` in `pyproject.toml`): no bare `print` outside the CLI,
  annotations on every function, imports sorted. `mypy --strict` covers `core`, `routing` and
  `skills`. Fix at the source rather than adding a `noqa`; a real exception gets the narrowest
  scope and a one-line reason.
- No global mutable state except `AdapterState`, which is passed by reference and owned by the
  model it was injected into.
- Every experiment writes a `RunRecord`. A number that is not in a run record does not go in a
  table, a doc, or a message.
- `train.seed` and `eval.seed` are never equal. The config rejects it.
- A skill is data and a grader. A grader never loads a model, touches a GPU, or calls a network.
- Errors are `BijouError` subclasses from `core.types`. No bare `Exception`, no `assert` for
  control flow outside tests.
- Never edit `third_party/`. Update by moving the submodule; upstream behaviour we depend on gets
  a parity test in `tests/`.
- Comment style: see the `comment-style` skill. Plain ASCII, state what the code does, no
  rationale — the why belongs in the commit message and in `docs/decisions/`.
- Commit messages: imperative subject at most 72 chars, body explains why.

## Skills

`.claude/skills/README.md` lists them. Use `test-driven-development` for features and fixes,
`systematic-debugging` for bugs, `verification-before-completion` before saying anything is done,
`comment-style` when writing comments, `adr` when a decision needs recording, `code-quality` for
cleanup passes.

## Out of scope

See "Out of scope" in `docs/ROADMAP.md`. The agent harness — planner, sub-agents, computer use,
MCP, browser sessions, scheduling — is not built here. Do not build toward it without an explicit
decision recorded in `docs/decisions/`.
