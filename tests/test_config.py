"""Contradictory settings are rejected at load, not clamped at use."""

import pytest

from bijou.core.config import Config
from bijou.core.types import ConfigError


def test_defaults_load():
    assert Config().adapter.rank == 16


def test_lm_head_target_rejected():
    with pytest.raises(ConfigError, match="weight-tied"):
        Config(adapter={"targets": ["lm_head"]})


def test_block_length_must_divide_gen_length():
    with pytest.raises(ConfigError, match="multiple of"):
        Config(sampling={"gen_length": 100, "block_length": 32})


def test_train_and_eval_seeds_must_differ():
    with pytest.raises(ConfigError, match="eval split overlaps"):
        Config(train={"seed": 7}, eval={"seed": 7})


def test_train_samples_must_fill_one_batch():
    with pytest.raises(ConfigError, match="one train.batch_size batch"):
        Config(train={"train_samples": 4, "batch_size": 16})


def test_env_override(monkeypatch):
    monkeypatch.setenv("BIJOU_ADAPTER__RANK", "8")
    assert Config().adapter.rank == 8
