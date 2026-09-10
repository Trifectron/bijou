---
name: experiment-hygiene
description: The rules for producing a number in this repo. Use when running an experiment, writing a result into a doc or a table, or comparing conditions.
---

# Experiment hygiene

## Every number comes from a run record

`bijou.core.runs.RunRecord` captures the resolved config, the git SHA, the environment, the input
digests and the scores. A number that is not in a run record does not go in a table, a doc, a
commit message, or a message to a person. Cite the `run_id`.

## The baseline is the tuned prompt

An adapter is compared against a well-tuned prompt on the same base model, not against zero-shot.
Full fine-tuning of the same skill is the upper bound, and it is affordable at this scale, so
there is no excuse for omitting it.

## Report the damage, not just the win

An adapter that wins its own eval while degrading the other skills' evals is not modular. Every
condition is scored against every skill. Report the whole row.

## Seeds

`train.seed` and `eval.seed` are never equal; the config rejects it. A difference inside seed
variance is not a result. Run more seeds before reporting a margin.

## Kill criteria are honoured

`docs/ROADMAP.md` states what result ends a step. When a kill criterion is met, say so plainly and
propose the negative result. Do not reach for a variant that rescues the hypothesis without
recording that the original was falsified.
