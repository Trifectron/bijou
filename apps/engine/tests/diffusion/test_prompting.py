"""The tuned-prompt baseline: rendering, demonstrations, and choosing on a dev split."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from engine.runtime.evaluate import prepare  # noqa: E402
from engine.runtime.prompting import (  # noqa: E402
    candidates,
    demonstrations,
    render,
    tune,
    tuned_eval_split,
)
from engine.skills import json_extract as je  # noqa: E402


def test_zero_shot_with_the_first_instruction_is_the_default_prompt():
    sample = je.generate(1, seed=5)[0]
    assert render(sample, je.INSTRUCTIONS[0], []).prompt == sample.prompt


def test_demonstrations_come_before_the_input(cfg):
    sample = je.generate(1, seed=5)[0]
    demos = demonstrations(cfg, je, 2)
    prompt = render(sample, je.INSTRUCTIONS[1], demos).prompt
    assert prompt.startswith(je.INSTRUCTIONS[1])
    assert all(d.target in prompt for d in demos)
    assert prompt.endswith(sample.meta["text"])
    assert sample.target not in prompt


def test_demonstrations_are_the_head_of_the_train_split(cfg):
    assert demonstrations(cfg, je, 3) == je.generate(10, cfg.train.seed)[:3]
    assert demonstrations(cfg, je, 0) == []


def test_candidates_cross_instructions_with_shots(cfg):
    pairs = {(c.instruction, c.shots) for c in candidates(cfg, je)}
    assert len(pairs) == len(je.INSTRUCTIONS) * len(cfg.prompting.shots)


def test_tuning_picks_one_candidate_and_renders_the_eval_split(cfg, make_backend):
    backend, state = prepare(cfg, [], backend=make_backend(cfg))
    choice = tune(cfg, backend, state, "json_extract")
    assert (choice.instruction, choice.shots) in {
        (c.instruction, c.shots) for c in candidates(cfg, je)
    }
    split = tuned_eval_split(cfg, "json_extract", choice)
    assert len(split) == cfg.eval.eval_samples
    assert all(s.prompt.startswith(je.INSTRUCTIONS[choice.instruction]) for s in split)
