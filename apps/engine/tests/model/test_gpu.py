"""The configured base checkpoint on the configured device. Run with just test-gpu.

Needs the checkpoint on disk (just checkpoints) and, when backend.device is cuda,
a GPU. Both are reported as skips with a reason rather than failures.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine.backends.nanodiff import NanoDiffBackend  # noqa: E402
from engine.core.config import load  # noqa: E402
from engine.core.types.model import GenerationRequest  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def backend():
    cfg = load()
    if cfg.backend.device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip(f"backend.device is {cfg.backend.device} and no GPU is visible")
    backend = NanoDiffBackend(cfg)
    if backend.checkpoint_path is None or not backend.checkpoint_path.exists():
        pytest.skip(f"no base checkpoint at {backend.checkpoint_path}; run: just checkpoints")
    backend.build()
    return backend


def test_the_checkpoint_answers_an_instruction(backend):
    cfg = backend.cfg.sampling
    out = backend.generate(
        GenerationRequest(
            prompt="What is the capital of France?",
            gen_length=cfg.gen_length,
            steps=cfg.steps,
            block_length=cfg.block_length,
        )
    )
    assert out.strip()
