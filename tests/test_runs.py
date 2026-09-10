"""A run record is immutable and carries enough to reproduce the run."""

import pytest

from bijou.core.runs import RunError, RunRecord, load_records


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
