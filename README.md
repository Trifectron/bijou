<h1 align="center">Bijou</h1>

<p align="center">
  <em>Modular capability deltas for masked diffusion language models.</em>
</p>

<p align="center">
  <a href="https://github.com/Trifectron/bijou/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Trifectron/bijou/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg">
</p>

---

Bijou tests one question:

> Can a masked diffusion language model carry **modular, composable, trajectory-routable**
> capabilities as detachable weight deltas, rather than encoding them in prompts or in a single
> monolithic fine-tune?

This is a research repo with kill criteria, not a product. `docs/ROADMAP.md` states the order of
the experiments and the result that ends each one.

## Quick start

```bash
git clone --recurse-submodules https://github.com/Trifectron/bijou
cd bijou
just bootstrap        # .env, git hooks, submodule, dependencies
just check            # the gate: format, lint, layering, types, tests
```

`just bootstrap` installs no torch, and `just check` runs in about a second. That is deliberate:
skills, graders, schedules and configuration are all testable without the model stack, so the
gate runs on every commit instead of on a GPU box occasionally.

```bash
just setup-train      # add CUDA torch when you have a GPU
just checkpoints      # the base model, from the Hugging Face Hub
just skill train json_extract
just evaluate
```

## The idea

A base model stays frozen. Named low-rank deltas attach to it, train one at a time on narrow
auto-gradable skills, and activate in combinations and at chosen points along the denoising
trajectory.

```
progress = 0.0 (all masked)  ------------------------->  1.0 (all committed)

  [ plan ]
          [ ------ domain ------ ]
                                  [ verify ]
```

An autoregressive model emits left to right, so an adapter is either on or off for the whole
generation. A diffusion LM generates by iterative global refinement, which makes *when* along the
trajectory a delta is active a real, tunable axis. That is the project's whole bet: if
phase-routed adapters do not beat statically applied ones, the substrate is buying nothing and
this should be a LoRA-composition project on an autoregressive model instead.

## The three questions

| # | Question | Kill criterion |
|---|---|---|
| 1 | Do adapters beat a tuned prompt on the same base model? | No skill wins outside seed variance |
| 2 | Do independently trained adapters compose without interfering? | Composition degrades and orthogonality-aware training does not recover it |
| 3 | Does phase routing beat static application? | The two are indistinguishable |

The baseline for question 1 is a *tuned prompt*, not zero-shot, and full fine-tuning of the same
skill is the upper bound. Both are cheap at this scale, so there is no excuse for omitting either.

## Layout

```
bijou/core          types, protocols, config, determinism, run records
bijou/adapters      the AdapterSite seam: LoRA injection, weights on disk
bijou/routing       the ActivationPolicy seam: phase schedules and the router
bijou/skills        one module per skill: generate(n, seed), grade(sample, output)
bijou/backends      the only package importing third_party/nanoDiff
bijou/runtime       train one adapter, score one condition
bijou/experiments   the matrix runner and the bijou CLI
```

**The dependency rule.** A package imports one below it, never a sibling:

```
experiments -> runtime -> {adapters, routing, skills, backends} -> core
```

Five `import-linter` contracts enforce it, run by `scripts/check-deps.sh` in `just check`, the
pre-commit hook, and CI. Two of them exist for specific reasons: `bijou.skills` may not import
torch, which is what keeps graders fast enough to sit inside the gate; and only `bijou.backends`
may import the vendored model, which is what makes a larger base model a sibling module rather
than a rewrite. `docs/ARCHITECTURE.md` explains the rest.

## The three seams

Everything else in the repo is infrastructure for these.

| Seam | Question it answers |
|---|---|
| `AdapterSite` | How does a delta attach to one frozen module? |
| `ActivationPolicy` | What is live at denoising step *t*? |
| `Grader` | How is a skill's output scored? |

`Backend` is the fourth interface, and exists so nanoDiff's internals are touched in exactly one
package.

## Console

```bash
just console          # or: just cli, bijou console
```

A terminal UI over every recipe: pick one on the left, press enter, and its output streams on
the right. The status bar shows GPU memory and which processes hold it, whether the base
checkpoint is on disk, how many adapters and full fine-tunes are trained, the last run record,
and the git SHA.

```
 bijou  NORMAL  ● gpu 5.4/6.0G (2x llama-server)  ● nanodiff-150m-sft-alpaca  adapters 0/1  full 0/1
╭ units ──────────────────────────╮╭ skill train json_extract · running · 3m05s · 812 lines · follow ─╮
│ setup                           ││ 20:42:52 $ just skill train json_extract                        │
│ ○ doctor                        ││ 20:42:55 ...                                                     │
│ gate                            ││                                                                  │
│ ✓ check                         ││                                                                  │
│ train                           ││                                                                  │
│ ● skill train json_extract      ││                                                                  │
│ evaluate                        ││                                                                  │
│ ○ evaluate                      ││                                                                  │
╰─────────────────────────────────╯╰──────────────────────────────────── LoRA adapter on json_extract ╯
 j/k move  ⏎ start/stop  x stop  r restart  h/l units/logs  / search  : command  ? help  q quit
```

| Key | Does |
|---|---|
| `j` `k` `gg` `G` `ctrl+d` `ctrl+u` | move, or scroll the focused pane; `G` on logs resumes following |
| `enter` / `s`, `x`, `r` | start or stop, stop, restart the selected unit |
| `h` `l` `tab` | focus units, logs |
| `/` then `n` `N` | search the selected unit's logs |
| `C` | clear the selected unit's logs |
| `:` | command line: `:start <unit>`, `:stop <unit>`, `:restart <unit>`, `:clear`, `:help`, `:q` |
| `?` | help |
| `q` | quit; running units are stopped |

Anything else typed after `:` runs as a just recipe, so `:skill sample json_extract -n 2` shows up
under **ad-hoc**. Each unit runs in its own process group, so a stop reaches uv and python too.
Every line is also appended to `.bijou/logs/<unit>.log`; `[console]` in `bijou.toml` sets the
buffer size, the log directory and the status interval.

## CLI

```
bijou config [section]           the resolved configuration
bijou skill list                 known skills and whether each has a trained adapter
bijou skill sample <name> -n 3   generated samples, no model loaded
bijou skill grade <name> "..."   run a grader against one sample
bijou skill train <name>         train an adapter, or --full-finetune for the upper bound
bijou evaluate                   score the composition matrix
bijou run list | show <id>       run records
```

Everything above the `train` line works without torch installed.

## Reproducibility

Every train and evaluate run writes an immutable record under `runs/` holding the resolved
config, the git SHA, the torch and GPU versions, the input digests and the scores. Overwriting one
is refused. A number that is not in a run record does not go in a table, a doc, or a message —
see the `experiment-hygiene` skill.

Configuration is `bijou.toml`, committed, overridden by `BIJOU_<SECTION>__<KEY>` in the
environment. Contradictory settings are rejected at load rather than clamped at use: a target
naming `lm_head` (weight-tied to `tok_emb`), a block length that does not divide the generation
length, or equal train and eval seeds.

## Substrate

[nanoDiff](https://github.com/BY571/nanoDiff), pinned as a submodule. LLaDA-style masked
diffusion, 50M / 150M / 350M checkpoints, about 3k lines.

Chosen for iteration speed over capability. The experiments are matrices — every adapter subset
against every skill eval, across seeds — which is hours at 150M and a funding application at 7B.
The cost is that no model this size can run an agent harness, so the harness is out of scope here.
See `docs/decisions/0001-nanodiff-as-the-substrate.md`.

## Naming

These are **adapters** or **capability deltas**, never "SLoRA" —
[S-LoRA](https://arxiv.org/abs/2311.03285) is an existing serving system and the collision
misleads. "Skill" means the module in `bijou/skills`: data plus a grader.

## Status

Scaffold. The CPU gate is green. The model-stack paths have not been run against a real
checkpoint, and roadmap step 0 is the parity test that validates them: the reimplemented
denoising loop must match upstream `generate` token for token with no adapter active.

Contributing: read [AGENTS.md](AGENTS.md).

## License

MIT.
