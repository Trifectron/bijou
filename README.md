# Bijou

Bijou tests one question:

> Can a masked diffusion language model carry **modular, composable, trajectory-routable**
> capabilities as detachable weight deltas, rather than encoding them in prompts or in a
> single monolithic fine-tune?

That is a research question, not a product. Read the scope section before adding anything.

## Substrate

- **Base model:** [nanoDiff](https://github.com/BY571/nanoDiff), vendored at `third_party/nanoDiff`
  (pinned @ `312a9e7`). LLaDA-style masked diffusion, LLaMA-style bidirectional transformer,
  50M / 150M / 350M checkpoints, ~3k LOC total.
- **Why nanoDiff over LLaDA-8B or Dream-7B:** the whole point of the project is to run a
  full composition matrix many times over. At 150M that is hours; at 7B it is a grant.
  nanoDiff also exposes the two things we need to modify — the SFT masking (`nanodiff/sft.py`)
  and the per-step denoising loop (`nanodiff/sampler.py`) — as plain readable functions
  rather than behind a framework.
- **Adapters:** `bijou/lora.py`. Targets `attn.qkv`, `attn.proj`, `mlp.w{1,2,3}`.
  `lm_head` is excluded on purpose — it is weight-tied to `tok_emb`.

At this scale LoRA saves no memory. It is used for **modularity**, not efficiency, which is
why full fine-tuning of the same skill is a required baseline (see `docs/experiments.md`).

## Scope

**In scope**

- Training adapters for narrow, auto-gradable skills on a fixed base checkpoint.
- Measuring specialisation, cross-task damage, and interference under composition.
- Routing different adapters to different phases of the denoising trajectory.

**Explicitly out of scope, for now**

- The agent harness — planner, sub-agents, computer use, MCP, browser sessions, scheduled
  long-horizon tasks. A 350M model cannot run it, and building it here would mean measuring
  adapter effects through a broken agent loop. It belongs in a separate repo on a capable
  base model, and only if the results below come back positive.
- A learned skill router. There is nothing to route until composition is shown to work.
- Adapter serving/paging systems work. Solved elsewhere (S-LoRA); not a contribution.

## Why diffusion at all

One reason, and it should be stated honestly because it is the project's whole bet:
a diffusion LM's generation is an iterative global refinement, not a left-to-right emission.
That makes *when* along the trajectory an adapter is active a real, tunable axis that has no
autoregressive counterpart. If phase-routed adapters do not beat statically-applied ones,
the diffusion substrate is buying nothing and this should be a LoRA-composition project on
an AR model instead.

## Naming

Do not call these "SLoRA" — [S-LoRA](https://arxiv.org/abs/2311.03285) is an existing MLSys
serving system and the collision misleads. They are **adapters** or **capability modules**.
"Skill" is reserved for the manifest-level object (`bijou/skills.py`): adapter + base
checkpoint + dataset + eval + recorded scores.

## Status

Scaffold only. `bijou/lora.py` and `bijou/phase.py` are written against the upstream source
but have **not been executed** — they were authored in a container without torch. First task
is `docs/experiments.md` step 0.

## Layout

```
bijou/            adapter + phase-routing + manifest code
skills/           one directory per skill: manifest.toml (+ adapter weights, gitignored)
docs/             experiment plan
third_party/      nanoDiff submodule (pristine upstream)
```
