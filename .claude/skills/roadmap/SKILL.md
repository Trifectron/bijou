---
name: roadmap
description: Report progress against docs/ROADMAP.md. Use when asked what is done, what is next, or whether a step is finished.
---

# roadmap

Read `docs/ROADMAP.md`. Report the current step, what is checked off, and the single next
unchecked item.

A step is finished only when its kill criterion has been evaluated, not when its code exists. Say
which run records support the claim; `just runs` lists them. A step with no run record backing it
is not finished.

Do not mark an item complete in the doc without a run record or a passing test to point at.
