# Bijou

> Can a masked diffusion language model carry **modular, composable, trajectory-routable**
> capabilities as detachable weight deltas, rather than encoding them in prompts or in a single
> monolithic fine-tune?

That is a research question with kill criteria, not a product. `docs/ROADMAP.md` has the order and
the conditions under which each step ends the project.

## Quick start

```bash
just bootstrap     # .env, git hooks, submodule, deps
just check         # the gate: format, lint, layering, types, tests
just skill sample json_extract
```

`just check` runs on CPU with no model stack installed, in about a second. `just setup-train` adds
torch when you have a GPU box.

## What is here

A base model stays frozen. Named low-rank deltas attach to it, train one at a time on narrow
auto-gradable skills, and activate in combinations and at chosen points along the denoising
trajectory.

```
bijou/core          types, protocols, config, determinism, run records
bijou/adapters      the AdapterSite seam
bijou/routing       the ActivationPolicy seam
bijou/skills        one module per skill: generate(n, seed), grade(sample, output)
bijou/backends      the only package importing third_party/nanoDiff
bijou/runtime       train one adapter, score one condition
bijou/experiments   the matrix runner and the bijou CLI
```

The dependency rule is `experiments -> runtime -> {adapters, routing, skills, backends} -> core`,
enforced by `scripts/check-deps.sh` on every commit. `docs/ARCHITECTURE.md` explains why each edge
is where it is.

## Why diffusion

One reason, and it is the project's whole bet. A diffusion LM generates by iterative global
refinement, not left-to-right emission, so *when* along the trajectory an adapter is active is a
real, tunable axis with no autoregressive counterpart. If phase-routed adapters do not beat
statically applied ones, the substrate is buying nothing and this should be a LoRA-composition
project on an autoregressive model instead.

## Substrate

[nanoDiff](https://github.com/BY571/nanoDiff), pinned at `third_party/nanoDiff`. LLaDA-style
masked diffusion, 50M / 150M / 350M. Chosen for iteration speed over capability — see
`docs/decisions/0001-nanodiff-as-the-substrate.md` for what that costs.

## Naming

These are **adapters** or **capability deltas**, never "SLoRA" —
[S-LoRA](https://arxiv.org/abs/2311.03285) is an existing serving system and the collision
misleads. "Skill" means the module in `bijou/skills`: data plus a grader.

## Status

Scaffold. The CPU gate is green; the model-stack paths have not run against a real checkpoint.
Step 0 of `docs/ROADMAP.md` is the parity test.

Contributing: read `AGENTS.md`.
