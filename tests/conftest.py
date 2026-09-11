"""Fixtures. Everything here runs on CPU in under a second."""

from __future__ import annotations

import pytest

from bijou.core.config import Config


@pytest.fixture
def cfg() -> Config:
    """A configuration small enough to run on CPU."""
    return Config(
        backend={"checkpoint": "", "device": "cpu", "dtype": "float32", "compile": False},
        train={
            "prompt_len": 32,
            "response_len": 32,
            "batch_size": 2,
            "max_steps": 2,
            "train_samples": 8,
            "seed": 0,
        },
        sampling={"steps": 4, "gen_length": 16, "block_length": 8},
        eval={"eval_samples": 4, "seed": 1},
    )


@pytest.fixture
def tiny_model():
    """A 2-layer nanoDiff. Skipped when the model stack is not installed."""
    torch = pytest.importorskip("torch")
    from nanodiff.model import NanoDiff

    from bijou.backends.nanodiff import tiny_config

    torch.manual_seed(0)
    return NanoDiff(tiny_config())
