# 0002 — Build the agent here, in one engine with the skill bank

Accepted. Supersedes the "out of scope" consequence of 0001.

## Context

0001 put the agent harness out of scope: a 350M diffusion model cannot plan, call tools or drive a
browser, and measuring adapters through a broken agent loop would measure nothing.

The goal has moved from "does this work" to building the system in the v0 design: a user request
is planned into subagents, each subagent is equipped with the LoRA skills its work needs, acts
through tools and MCP servers, and every session is indexed so recurring work can be turned
into new skills.

## Decision

Build the agent in this repo, in the same package as the diffusion model, with the work split
between two models.

- **An LLM plans and selects.** An OpenAI-compatible chat model, llama-server by default, splits
  a request into steps, picks which skills each step equips, and drives each subagent's tool loop.
- **The diffusion model executes skills.** The skill bank loads the base model once with every
  trained LoRA attached. A subagent reaches it through one tool, `run_skill`, carrying exactly the
  skills, or the phase schedule, the selector picked.
- **One engine.** `apps/engine` holds the model, the bank, the agent, collection and the research
  matrix, with import contracts between its layers. The agent reaches the bank only through the
  `SkillRuntime` protocol: in this process by default, or over HTTP to `engine serve --bank` when
  the model runs on another machine.
- **Three apps.** `apps/engine`, `apps/evals` (golden cases against a serving engine, gated on a
  baseline, as in sparkyai) and `apps/cli` (the console). Apps never import each other.

A first cut put the agent in its own app talking to the model over HTTP. That duplicated the
wire types, the config and the error base, and forced a network hop on a single machine; merging
removed all three, and the protocol keeps the remote option.

## Why

The planner needs a capability no diffusion model we can run has. Splitting the roles keeps the
agent buildable today while the thing the project exists to test, LoRA skills on a diffusion
model and when along the trajectory they are live, stays on the request path rather than beside it.

## Cost

Building no longer waits on the research kill criteria. The roadmap's steps still decide what can
be claimed about adapters, but not whether the agent exists.

Agent quality is bounded by the chat model, and skill quality by a 150M base until a larger
diffusion backend exists. Agent answers are not evidence about adapters.

One package is larger; the layer contracts are what keep it from becoming one tangle.

## Reversal

If phase routing is indistinguishable from static application (roadmap step 3), the skill bank
can be replaced by an autoregressive LoRA server behind the same `SkillRuntime` protocol. The
agent does not change.

`0003` narrows the ways in that this record assumed: the agent is reached in a terminal, and the
skill bank always runs in the agent's process rather than optionally over HTTP.
