"""Training an adapter and scoring a condition, end to end on CPU.

Uses the stub tokenizer and a 2-layer model, so the loop is exercised without a
GPU or a base checkpoint. The numbers are meaningless; the wiring is the point.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine.core.runs import load_records  # noqa: E402
from engine.core.types.errors import ArtifactError  # noqa: E402
from engine.routing.phase import PhaseSchedule  # noqa: E402
from engine.runtime.evaluate import prepare, score_condition  # noqa: E402
from engine.runtime.matrix import conditions, run  # noqa: E402
from engine.runtime.train import train_adapter  # noqa: E402


@pytest.fixture
def local_cfg(cfg, tmp_path):
    """The CPU config, with every path inside a temporary directory."""
    return cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={
                    "runs": tmp_path / "runs",
                    "adapters": tmp_path / "adapters",
                    "base_checkpoints": tmp_path / "base",
                }
            )
        }
    )


def test_training_writes_an_adapter_and_a_run_record(local_cfg, make_backend):
    path = train_adapter(local_cfg, "json_extract", backend=make_backend(local_cfg))
    assert path.exists()

    (record,) = load_records(local_cfg.paths.runs)
    assert record.kind == "train"
    assert record.scores["trainable_params"] > 0
    assert record.scores["steps"] == local_cfg.train.max_steps
    assert "json_extract" in record.notes
    assert record.finished_at


def test_full_finetune_trains_more_parameters_than_an_adapter(local_cfg, make_backend):
    train_adapter(local_cfg, "json_extract", backend=make_backend(local_cfg))
    train_adapter(local_cfg, "json_extract", full_finetune=True, backend=make_backend(local_cfg))
    by_notes = {r.notes: r.scores["trainable_params"] for r in load_records(local_cfg.paths.runs)}
    adapter = by_notes["skill=json_extract full_finetune=False"]
    full = by_notes["skill=json_extract full_finetune=True"]
    assert full > adapter * 10


def test_scoring_a_condition_grades_every_sample(local_cfg, make_backend):
    train_adapter(local_cfg, "json_extract", backend=make_backend(local_cfg))
    backend, state = prepare(local_cfg, ["json_extract"], backend=make_backend(local_cfg))
    report, scores = score_condition(
        local_cfg,
        backend,
        state,
        "json_extract",
        condition="json_extract",
        schedule=PhaseSchedule.static("json_extract"),
    )
    assert report.total == local_cfg.eval.eval_samples == len(scores)
    assert 0.0 <= report.rate <= 1.0
    assert report.condition == "json_extract"


def test_untrained_model_scores_zero_but_still_grades(local_cfg, make_backend):
    backend, state = prepare(local_cfg, [], backend=make_backend(local_cfg))
    report, scores = score_condition(
        local_cfg, backend, state, "json_extract", "none", PhaseSchedule.static()
    )
    # A 2-layer untrained model emits nothing parseable; the grader must say so
    # rather than raise.
    assert report.passed == 0
    assert all(not s.passed and s.detail for s in scores)


def test_the_matrix_scores_every_condition(local_cfg, make_backend):
    train_adapter(local_cfg, "json_extract", backend=make_backend(local_cfg))
    train_adapter(local_cfg, "json_extract", full_finetune=True, backend=make_backend(local_cfg))
    reports = run(local_cfg, make_backend=make_backend)

    assert {r.condition for r in reports} == {
        "none",
        "json_extract",
        "tuned-prompt",
        "full:json_extract",
    }
    (record,) = [r for r in load_records(local_cfg.paths.runs) if r.kind == "eval"]
    assert {"adapter/json_extract", "full-finetune/json_extract"} <= set(record.inputs)
    assert "tuned-prompt/json_extract/shots" in record.scores


def test_the_matrix_names_the_missing_artifact(local_cfg, make_backend):
    with pytest.raises(ArtifactError, match="just skill train json_extract"):
        run(local_cfg, make_backend=make_backend)


def test_a_full_finetune_loads_as_a_base_checkpoint(local_cfg, make_backend):
    path = train_adapter(
        local_cfg, "json_extract", full_finetune=True, backend=make_backend(local_cfg)
    )
    assert path == local_cfg.full_finetune_path("json_extract")
    full_cfg = local_cfg.model_copy(
        update={"backend": local_cfg.backend.model_copy(update={"checkpoint": path.stem})}
    )
    make_backend(full_cfg).build()


def test_conditions_cover_every_subset():
    assert conditions(("a", "b")) == {
        "none": (),
        "a": ("a",),
        "b": ("b",),
        "a+b": ("a", "b"),
    }
