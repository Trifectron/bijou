# Roadmap

Ordered so each step can kill the project and the cheap ones come first. Do not build
infrastructure for step N+1 before step N reports.

## 0. Parity and plumbing

- [x] Package layout, dependency rule, gate, CLI
- [x] Adapter injection, weights on disk, phase schedules, one skill, run records
- [ ] Base checkpoints pulled and `just test-gpu` green
- [ ] The reimplemented denoising loop matches upstream `generate` token for token with no
      adapter active

Until parity holds, every downstream number is uninterpretable.

## 1. Do adapters beat prompting?

Null hypothesis is a tuned prompt on the same base model, not zero-shot.

- [ ] Skills: `json_extract`, `normalize`, `fn_call`
- [ ] Conditions per skill: base zero-shot, base tuned prompt, LoRA, full fine-tune
- [ ] Report each adapter's own score and its damage to the other skills' evals

Kill criterion: no skill beats a tuned prompt outside seed variance.

## 2. Does composition survive?

- [ ] The full matrix: every adapter subset against every skill eval
- [ ] A jointly fine-tuned model as the ceiling

Kill criterion: composition degrades and orthogonality-aware training does not recover it.
Publish the negative result; do not build the router.

## 3. Does phase routing beat static application?

The only diffusion-specific claim.

- [ ] Static, early/late split, and the reversed split on a task needing two skills
- [ ] Boundaries aligned to sampler blocks, or the cache off

Kill criterion: phase routing is indistinguishable from static. Then the diffusion substrate is
not earning its keep, and the honest response is to say so.

## Out of scope

- The agent harness: planner, sub-agents, computer use, MCP, browser sessions, scheduling
- A learned skill router
- Adapter serving, paging and batching systems work
- Any base model above 350M, until step 3 reports
