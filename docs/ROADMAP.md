# Roadmap

Two tracks. The research track is ordered so each step can kill the claim and the cheap ones come
first; it decides what can be said about adapters. The agent track builds the system in the v0
design (`docs/decisions/0002-*`); it does not wait on the research track, and its results are not
evidence about adapters.

## Research

### 0. Parity and plumbing

- [ ] Base checkpoints pulled and `just test-gpu` green

Until parity holds, every downstream number is uninterpretable.

### 1. Do adapters beat prompting?

Null hypothesis is a tuned prompt on the same base model, not zero-shot.

- [ ] Skills: `json_extract`, `normalize`, `fn_call`
- [ ] Report each adapter's own score and its damage to the other skills' evals

Kill criterion: no skill beats a tuned prompt outside seed variance.

### 2. Does composition survive?

- [ ] The full matrix: every adapter subset against every skill eval
- [ ] A jointly fine-tuned model as the ceiling

Kill criterion: composition degrades and orthogonality-aware training does not recover it.
Publish the negative result.

### 3. Does phase routing beat static application?

The only diffusion-specific claim.

- [ ] Static, early/late split, and the reversed split on a task needing two skills

Kill criterion: phase routing is indistinguishable from static. Then the diffusion substrate is
not earning its keep; the skill bank moves to an autoregressive LoRA server behind the same
`SkillRuntime` protocol (0002, Reversal).

## Agent

### A1. First live runs

- [ ] `just serve` against llama-server with `json_extract` trained
- [ ] `just evals run`, then commit the first `apps/evals/baseline.json`
- [ ] The browser through Playwright MCP on a read-only task

### A2. The skill loop, end to end

- [ ] One proposal from `just patterns --write`, approved, collected, trained, and equipped
- [ ] An eval case that needs the new skill, passing

### A3. A skill executor worth equipping

- [ ] A LLaDA or Dream backend as a sibling in `engine/backends`, run on a rented GPU
- [ ] The same eval suite against both backends, compared against the baseline

## Out of scope

- A trained skill router, until there are sessions to train one on
- Adapter paging, batching and multi-GPU serving
- Authenticated browser sessions and stored credentials
