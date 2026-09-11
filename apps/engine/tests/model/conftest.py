"""Fixtures. Everything here runs on CPU in under a second."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from engine.core.config import Config


class StubTokenizer:
    """One token per character, with a reserved end-of-text id."""

    eot_token = 50256

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 5000 + 100 for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr((t - 100) % 5000) for t in tokens)


@pytest.fixture
def cfg() -> Config:
    """A configuration small enough to run on CPU, with no base checkpoint."""
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
        prompting={"shots": (0, 1), "dev_samples": 2, "seed": 2},
    )


@pytest.fixture
def make_backend() -> Callable[[Config], Any]:
    """Builds a 2-layer nanoDiff backend with the stub tokenizer."""
    pytest.importorskip("torch")
    from engine.backends.nanodiff import NanoDiffBackend, tiny_config

    def build(cfg: Config) -> NanoDiffBackend:
        return NanoDiffBackend(cfg, nano=tiny_config(), tokenizer=StubTokenizer())

    return build


@pytest.fixture
def tiny_model():
    """A 2-layer nanoDiff. Skipped when the model stack is not installed."""
    torch = pytest.importorskip("torch")
    from nanodiff.model import NanoDiff

    from engine.backends.nanodiff import tiny_config

    torch.manual_seed(0)
    return NanoDiff(tiny_config())
