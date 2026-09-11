"""Collection: only approved specs, deduplicated examples, fixed splits, and a run record."""

import json

import pytest

from engine.collect.pipeline import collect, gather, split
from engine.collect.spec import Pair, SkillSpec, SpecError, list_specs, read_spec, write_spec
from engine.collect.teacher import parse, prompt
from engine.core.runs import load_records
from engine.skills import load


class CountingTeacher:
    """Writes numbered examples, repeating the first one every batch."""

    def __init__(self):
        self.calls = 0

    def write(self, spec, n, seen):
        self.calls += 1
        batch = [Pair(input="repeat", output="r")]
        batch += [Pair(input=f"q{self.calls}-{i}", output=f"a{i}") for i in range(n)]
        return batch


def spec(**overrides):
    fields = {
        "name": "echo",
        "description": "repeats things",
        "instruction": "Repeat it.",
        "approved": True,
    }
    return SkillSpec(**(fields | overrides))


@pytest.fixture
def local_cfg(cfg, tmp_path):
    return cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={"data": tmp_path / "data", "runs": tmp_path / "runs"}
            ),
            "collect": cfg.collect.model_copy(update={"examples": 20, "per_request": 5}),
        }
    )


def test_an_unapproved_spec_is_refused(local_cfg):
    with pytest.raises(SpecError, match="not approved"):
        collect(local_cfg, spec(approved=False), CountingTeacher())


def test_a_built_in_skill_cannot_be_collected_over(local_cfg):
    with pytest.raises(SpecError, match="built-in"):
        collect(local_cfg, spec(name="json_extract"), CountingTeacher())


def test_gathering_keeps_the_specs_pairs_and_drops_duplicate_inputs(local_cfg):
    pairs = gather(local_cfg, spec(pairs=[Pair(input="seed", output="s")]), CountingTeacher(), 12)
    inputs = [p.input for p in pairs]
    assert inputs[0] == "seed"
    assert len(inputs) == len(set(inputs)) == 12
    assert inputs.count("repeat") == 1


def test_the_split_is_fixed_by_the_collect_seed(local_cfg):
    pairs = [Pair(input=str(i), output=str(i)) for i in range(20)]
    first, second = split(local_cfg, pairs), split(local_cfg, pairs)
    assert first == second
    assert len(first["eval"]) == 4 and len(first["dev"]) == 2 and len(first["train"]) == 14


def test_collecting_writes_a_trainable_skill_and_a_run_record(local_cfg):
    out = collect(local_cfg, spec(), CountingTeacher())
    skill = load("echo", local_cfg.paths.data)
    assert skill.DESCRIPTION == "repeats things"
    assert skill.generate(100, 0, split="train")
    record = load_records(local_cfg.paths.runs)[-1]
    assert record.kind == "collect"
    assert record.scores["collected"] == 20.0
    assert (local_cfg.paths.runs / record.run_id / "artifact").read_text() == str(out)
    with pytest.raises(SpecError, match="exists"):
        collect(local_cfg, spec(), CountingTeacher())


def test_spec_files_round_trip_and_refuse_to_be_replaced(tmp_path):
    path = tmp_path / "echo.json"
    write_spec(spec(approved=False), path)
    assert read_spec(path).name == "echo"
    with pytest.raises(SpecError, match="exists"):
        write_spec(spec(), path)
    (tmp_path / "broken.json").write_text("{}")
    listed = dict((p.name, s) for p, s in list_specs(tmp_path))
    assert listed["broken.json"] is None and listed["echo.json"].approved is False


def test_a_spec_name_must_be_snake_case():
    with pytest.raises(ValueError, match="snake case"):
        spec(name="Not Valid")


def test_teacher_replies_parse_and_malformed_items_are_dropped():
    content = json.dumps(
        {"examples": [{"input": "a", "output": "b"}, {"input": "", "output": "x"}, "junk"]}
    )
    payload = {"choices": [{"message": {"content": content}}]}
    assert parse(payload) == [Pair(input="a", output="b")]
    assert parse({"choices": [{"message": {"content": "not json"}}]}) == []
    assert parse({}) == []


def test_the_teacher_prompt_carries_the_spec():
    text = prompt(spec(examples=["seen input"]), 4, [Pair(input="x", output="y")])
    assert "Repeat it." in text and "seen input" in text and "Write 4 new examples" in text
