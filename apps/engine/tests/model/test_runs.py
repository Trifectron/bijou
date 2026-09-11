"""A run record is immutable and carries enough to reproduce the run."""

import pytest

from engine.core.determinism import git_sha
from engine.core.runs import RunError, RunRecord, load_records


def test_record_round_trips(tmp_path):
    record = RunRecord.start("eval", {"eval": {"seed": 1}})
    record.finish(accuracy=0.5)
    record.write(tmp_path)
    (loaded,) = load_records(tmp_path)
    assert loaded.scores["accuracy"] == 0.5
    assert loaded.config == {"eval": {"seed": 1}}


def test_record_captures_the_git_sha():
    assert RunRecord.start("train", {}).env["git_sha"]


def test_overwriting_a_run_is_refused(tmp_path):
    record = RunRecord.start("train", {}, run_id="fixed")
    record.write(tmp_path)
    with pytest.raises(RunError, match="already exists"):
        RunRecord.start("train", {}, run_id="fixed").write(tmp_path)


def test_two_runs_in_the_same_second_do_not_collide(tmp_path):
    first = RunRecord.start("train", {})
    second = RunRecord.start("train", {})
    assert first.run_id != second.run_id
    first.write(tmp_path)
    second.write(tmp_path)
    assert len(load_records(tmp_path)) == 2


def test_git_sha_is_quiet_outside_a_repository(tmp_path, monkeypatch, capfd):
    monkeypatch.chdir(tmp_path)
    assert git_sha() == "unknown"
    assert capfd.readouterr().err == ""
