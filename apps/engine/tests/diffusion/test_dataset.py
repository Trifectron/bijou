"""Dataset skills: collected data on disk behaves like a built-in skill, with fixed splits."""

import json

import pytest

from engine.skills import UnknownSkill, load, names
from engine.skills.dataset import DatasetError, DatasetSkill


def write_skill(data, name="greet", rows=None):
    directory = data / "skills" / name
    directory.mkdir(parents=True)
    (directory / "skill.json").write_text(
        json.dumps({"name": name, "description": "says hello", "instruction": "Greet them."})
    )
    rows = rows or {
        "train": [("Ana", "Hello, Ana."), ("Ben", "Hello, Ben."), ("Chen", "Hello, Chen.")],
        "dev": [("Dara", "Hello, Dara.")],
        "eval": [("Eli", "Hello, Eli."), ("Farah", "Hello, Farah.")],
    }
    for split, pairs in rows.items():
        lines = [json.dumps({"input": i, "output": o}) for i, o in pairs]
        (directory / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    return directory


def test_a_collected_skill_is_listed_after_the_built_in_ones(tmp_path):
    write_skill(tmp_path)
    assert names(tmp_path) == ("json_extract", "greet")
    assert names(None) == ("json_extract",)


def test_a_collected_skill_loads_by_name_and_unknown_names_list_it(tmp_path):
    write_skill(tmp_path)
    skill = load("greet", tmp_path)
    assert skill.NAME == "greet"
    assert skill.DESCRIPTION == "says hello"
    with pytest.raises(UnknownSkill, match="greet"):
        load("nope", tmp_path)


def test_splits_never_overlap_whatever_the_seed(tmp_path):
    skill = load("greet", write_skill(tmp_path).parents[1])
    for seed in range(5):
        train = {s.meta["text"] for s in skill.generate(10, seed, split="train")}
        evals = {s.meta["text"] for s in skill.generate(10, seed, split="eval")}
        assert not train & evals


def test_generation_is_seed_stable_and_capped_by_the_split(tmp_path):
    skill = load("greet", write_skill(tmp_path).parents[1])
    assert skill.generate(3, 7, split="train") == skill.generate(3, 7, split="train")
    assert len(skill.generate(100, 0, split="eval")) == 2
    sample = skill.generate(1, 0, split="dev")[0]
    assert sample.prompt == "Greet them.\n\nDara"


def test_text_grading_ignores_case_and_whitespace(tmp_path):
    skill = load("greet", write_skill(tmp_path).parents[1])
    sample = skill.generate(1, 0, split="dev")[0]
    assert skill.grade(sample, "  hello,   DARA. ").passed
    partial = skill.grade(sample, "Hello there")
    assert not partial.passed
    assert 0 < partial.value < 1


def test_json_targets_grade_by_value(tmp_path):
    rows = {s: [("x", json.dumps({"a": 1, "b": 2}))] for s in ("train", "dev", "eval")}
    skill = load("jsonish", write_skill(tmp_path, "jsonish", rows).parents[1])
    sample = skill.generate(1, 0, split="eval")[0]
    assert skill.grade(sample, '{"b": 2, "a": 1}').passed
    half = skill.grade(sample, '{"a": 1, "b": 3}')
    assert not half.passed and half.value == pytest.approx(0.5)


def test_a_missing_split_and_a_bad_line_are_named(tmp_path):
    directory = write_skill(tmp_path)
    (directory / "dev.jsonl").unlink()
    with pytest.raises(DatasetError, match="dev.jsonl is missing"):
        DatasetSkill.read(directory)
    (directory / "dev.jsonl").write_text("not json\n")
    with pytest.raises(DatasetError, match="dev.jsonl:1"):
        DatasetSkill.read(directory)


def test_the_directory_and_the_name_must_agree(tmp_path):
    directory = write_skill(tmp_path)
    (directory / "skill.json").write_text(
        json.dumps({"name": "other", "description": "d", "instruction": "i"})
    )
    with pytest.raises(DatasetError, match="lives in"):
        DatasetSkill.read(directory)
