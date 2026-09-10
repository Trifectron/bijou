# 0001 — nanoDiff as the substrate, not LLaDA or Dream

Accepted.

## Context

The project needs a masked diffusion LM to attach adapters to. The credible general-purpose
options are LLaDA-8B and Dream-7B. nanoDiff is a 3k-line reimplementation with 50M, 150M and 350M
checkpoints.

## Decision

Build on nanoDiff, pinned as a submodule.

## Why

The experiments are matrices. Every adapter subset against every skill eval, repeated across
seeds, is hours at 150M and a funding application at 7B.

The two things the project must modify are plain functions in nanoDiff: the SFT masking in
`nanodiff/sft.py` and the per-step denoising loop in `nanodiff/sampler.py`. On a 7B HuggingFace
model the same modification is a week of plumbing.

## Cost

A 350M model cannot run an agent harness. Accepting this decision means the harness is out of
scope for this repo, which `docs/ARCHITECTURE.md` records.

LoRA at this scale saves no memory, so every adapter result is compared against a full fine-tune
of the same skill as the upper bound.

## Reversal

`bijou/backends` is the only package importing the vendored model. A larger base model is a
sibling module there.
