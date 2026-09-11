"""Skills are data and graders. These run without torch, in milliseconds."""

import json

import pytest

from engine.core.types.model import Sample
from engine.skills import KNOWN, UnknownSkill, load
from engine.skills import json_extract as je


def test_unknown_skill_named():
    with pytest.raises(UnknownSkill, match="known:"):
        load("nope")


def test_generation_is_seed_stable():
    assert je.generate(5, seed=1) == je.generate(5, seed=1)
    assert je.generate(5, seed=1) != je.generate(5, seed=2)


def test_target_grades_perfect():
    sample = je.generate(1, seed=1)[0]
    score = je.grade(sample, sample.target)
    assert score.passed and score.value == 1.0


def test_unparseable_output_fails_with_a_reason():
    sample = je.generate(1, seed=1)[0]
    score = je.grade(sample, "not json at all")
    assert not score.passed
    assert "unparseable" in score.detail


def test_partial_credit_is_reported():
    sample = je.generate(1, seed=1)[0]
    partial = json.loads(sample.target)
    partial["city"] = "Nowhere"
    score = je.grade(sample, json.dumps(partial))
    assert not score.passed
    assert score.value == pytest.approx(0.75)


@pytest.mark.parametrize("name", KNOWN)
def test_every_skill_supports_the_tuned_prompt_baseline(name):
    skill = load(name)
    assert skill.INSTRUCTIONS
    assert all(s.meta.get("text") for s in skill.generate(3, seed=1))


@pytest.mark.parametrize("name", KNOWN)
def test_every_skill_describes_itself_for_the_selector(name):
    skill = load(name)
    assert name == skill.NAME
    assert len(skill.DESCRIPTION) > 20
    assert "{instruction}" in skill.PROMPT and "{text}" in skill.PROMPT


def test_non_object_output_fails():
    sample = Sample(id="x", prompt="p", target=json.dumps({"name": "a"}))
    assert not je.grade(sample, "[1, 2]").passed
