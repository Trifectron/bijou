"""The configured base checkpoint on the configured device. Run with just test-gpu.

Needs the checkpoint on disk (just checkpoints) and, when backend.device is cuda,
a GPU. Both are reported as skips with a reason rather than failures.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bijou.backends.nanodiff import NanoDiffBackend  # noqa: E402
from bijou.core.config import load  # noqa: E402
from bijou.core.types import GenerationRequest  # noqa: E402

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


def test_the_checkpoint_matches_upstream_generate(backend):
    from nanodiff.sampler import generate as upstream_generate

    cfg = backend.cfg.sampling
    prompt = torch.tensor(
        [backend.prompt_ids("Name three primary colors.")], device=backend.nano.device
    )
    request = GenerationRequest(
        prompt="", gen_length=cfg.gen_length, steps=cfg.steps, block_length=cfg.block_length
    )
    ours = backend.denoise(prompt, request)
    with backend.autocast():
        theirs = upstream_generate(
            backend.model,
            prompt,
            gen_length=cfg.gen_length,
            steps=cfg.steps,
            block_length=cfg.block_length,
        )
    assert torch.equal(ours, theirs)


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
