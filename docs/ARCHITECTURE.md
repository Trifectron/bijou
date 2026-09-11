# Architecture

## What this is

A base masked diffusion LM stays frozen. Named low-rank deltas attach to it, are trained one at a
time on narrow auto-gradable skills, and are activated in combinations and at chosen points along
the denoising trajectory. The repo exists to measure whether those deltas stay modular,
composable and trajectory-routable.

## Packages

```
                     experiments        matrix runner, CLI
                          |
                       runtime          train one adapter, score one condition
                          |
        +--------+--------+--------+----------+
        |        |        |        |          |
     adapters  routing  skills  backends      |
        |        |        |        |          |
        +--------+--------+--------+----------+
                          |
                        core          types, protocols, config, determinism, runs
```

A package imports one below it, never a sibling. Enforced by `scripts/check-deps.sh` against the
contracts in `pyproject.toml`.

`core` holds what everything builds on and nothing that does work: the data that crosses a
boundary as a value, the interfaces, configuration, seeding, and run records.

`adapters` owns the injection sites. `LoRALinear` wraps a frozen `nn.Linear` and holds a
name-keyed bank of deltas. Activation is read from an `AdapterState` held by reference, so
changing the live set is a dict assignment rather than a module walk — which is what makes
per-denoising-step routing affordable.

`routing` owns schedules over the trajectory. It never imports `adapters`; `AdapterState` lives in
`core` so a schedule is testable with no adapter implementation loaded.

`skills` is one module per skill, each exposing `generate(n, seed)` and `grade(sample, output)`.
It imports no torch. That is deliberate: graders run in milliseconds on CPU, so the eval suite
runs on every commit rather than on a GPU box occasionally.

`backends` is the vendor layer. It is the only package that imports `third_party/nanoDiff`, and it
holds everything nanoDiff-shaped: `Config` construction, the SFT encoder, the masking objective,
and the reverse process. Moving to LLaDA or Dream means writing a sibling module.

`runtime` composes them: training, scoring one condition, and the tuned-prompt baseline, which
picks a skill's instruction and worked examples on a dev split with its own seed. `experiments`
runs the matrix and owns the CLI, and nothing imports it. `eval.conditions` selects the matrix
rows: adapter subsets (the empty subset is zero-shot), the tuned prompt, and one full fine-tune
per skill.

## Invariants

Adapter targets are named by module suffix and configured. `lm_head` is rejected at config load
because `tie_embeddings` ties it to `tok_emb`, so adapting the head would adapt the embedding
table.

A newly attached adapter is inert: `B` is zero-initialised, so activating an untrained adapter
cannot perturb the frozen base. `tests/test_adapters.py` asserts bit-identical logits.

The denoising loop lives in `backends`, reimplemented rather than reused, because upstream
`generate` takes no per-step hook and the `ActivationPolicy` seam needs one. It must match
upstream token for token when no adapter is active.

Generation runs with no prefix K/V cache. The cache is prefilled once per sampler block and reused
across that block's steps, so an adapter change mid-block leaves it describing weights that are no
longer live. `PhaseSchedule.validate_against_blocks` rejects a boundary that would do this.

Every train and evaluate run writes an immutable `RunRecord` holding the resolved config, the
environment, and the scores. Writing over an existing run is refused.

## What is not here

The agent harness. A 350M base cannot run a planner, tool use, or a browser session, and building
one here would mean measuring adapter effects through a broken agent loop. It belongs in a
separate repo on a capable base model, and only if the experiments come back positive.

A learned skill router. There is nothing to route until composition is shown to work.

Adapter serving and paging. Solved elsewhere; not a contribution.
