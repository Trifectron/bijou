# Experiment plan

The ordering is deliberate: each step can kill the project, and the cheap ones come first.
Do not build infrastructure for step N+1 before step N reports.

## Step 0 — parity and plumbing

Prove the adapter machinery is inert before trusting any delta it produces.

1. `inject(model)` on a 150M checkpoint, then generate with no adapters attached.
   Output must be **bit-identical** to upstream `nanodiff.sampler.generate`.
2. `add_adapter(model, "noop")` and activate it. Because `_LoRABranch.B` is zero-initialised,
   output must still be bit-identical. If it is not, the injection is wrong.
3. `freeze_base(model, "noop")` and assert the trainable count equals the analytic
   LoRA parameter count for the target set.

Until all three pass, every downstream number is uninterpretable.

## Step 1 — do adapters beat prompting? (RQ1)

The null hypothesis is a well-tuned prompt on the same base model. Not zero-shot. Most
"cognitive role" skills (planning, verification, research) will lose this comparison, which
is why the first skills are narrow and format-heavy instead.

Skills v0 — synthetic, auto-gradable, mutually distinct:

| skill | task | grader |
|---|---|---|
| `json_extract` | free text -> fixed JSON schema | `json.loads` + exact schema/value match |
| `normalize` | dates/units/currencies -> canonical form | exact match |
| `fn_call` | request -> fixed function-call syntax | parse + argument match |

`fn_call` may be sliced from the Hermes-3 function-calling subset. Do **not** train on the
Hermes mixture as a whole: it is multi-turn ShareGPT against nanoDiff's fixed-width
single-turn `encode_sft_example`, most samples exceed `block_size=1024`, it is a generalist
blend that deliberately entangles the capabilities we are trying to separate, and at 150M
most of its content is above capability so the eval signal is noise.

Conditions per skill, all on the same base checkpoint:

- base, zero-shot
- base, tuned prompt  ← the real baseline
- LoRA adapter
- full fine-tune      ← the upper bound; affordable at this scale, so no excuse to skip it

Report the adapter's score **and** its cross-task damage: score on the other skills' evals.
An adapter that wins its own eval by destroying the others is not modular.

## Step 2 — does composition survive? (RQ2)

The load-bearing assumption, and the most likely failure. Independently trained LoRAs occupy
overlapping subspaces because nothing in training pushed them apart; degradation on summation
is the common outcome, not the exception.

Fill the full matrix — rows are active adapter sets, columns are the per-skill evals:

|            | json_extract | normalize | fn_call |
|---|---|---|---|
| none       | | | |
| A          | | | |
| B          | | | |
| C          | | | |
| A+B        | | | |
| A+C        | | | |
| B+C        | | | |
| A+B+C      | | | |
| joint FT   | | | |

`joint FT` — one model fine-tuned on all three at once — is the ceiling composition is
being measured against.

If A+B lands materially below both A and B on their own evals, the modular-skill premise is
false as stated. The follow-up is not to abandon it but to make composability a *training*
objective (orthogonality penalties between adapter subspaces, disjoint rank allocation,
routing-aware losses). That is a real result either way; it is also a different project than
the one in the diagram, and should be recognised as such.

## Step 3 — does phase routing beat static application? (RQ3)

The only diffusion-specific claim in the project.

Hook `PhaseRouter.at(step, total_steps)` into `nanodiff.sampler.generate`'s denoising loop,
immediately before each forward. Compare on a task needing more than one skill:

- static: all needed adapters active for the whole trajectory
- early/late split: adapter A over `[0.0, 0.5)`, adapter B over `[0.5, 1.0)`
- the reverse split — if order does not matter, the effect is not what we think it is

Mind the cache: `generate(use_cache=True)` prefills K/V per block and reuses it across that
block's steps, so an adapter change mid-block leaves stale cache. Either align phase
boundaries to block boundaries (`PhaseSchedule.validate_against_blocks`) or run with
`use_cache=False`.

A positive result here is the paper. A negative result means the diffusion substrate is not
earning its keep, and the honest response is to say so.

## Kill criteria

State these now, while they are still cheap to accept.

- Step 1: no skill beats a tuned prompt by a margin outside seed variance → adapters are the
  wrong mechanism at this scale.
- Step 2: composition degrades and orthogonality-aware training does not recover it →
  publish the negative result; do not build the router.
- Step 3: phase routing is indistinguishable from static → drop diffusion, or drop the
  routing claim.
