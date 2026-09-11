# Decisions

One record per decision that moved a boundary, changed an experiment, or foreclosed an option a
reasonable person would otherwise try. Records are immutable: a reversal is a new record that
supersedes the old one, and the old one gains a link forward.

| # | Decision | What it settles |
|---|---|---|
| [0001](0001-nanodiff-as-the-substrate.md) | nanoDiff as the substrate | which masked diffusion LM the skill bank is built on, and why it is vendored |
| [0002](0002-build-the-harness-over-the-skill-bank.md) | Build the harness over the skill bank | why the agent lives in this repo, and how the work splits between two models |
| [0003](0003-talk-to-the-agent-in-the-terminal.md) | Talk to the agent in the terminal | one way in: `engine chat`, no HTTP surface, the skill bank in the agent's process |
