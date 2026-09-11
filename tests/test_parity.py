"""Roadmap step 0: the reimplemented denoising loop matches upstream generate.

The CPU test runs the tiny random model in the gate. The gpu test runs the
configured base checkpoint and skips when it is absent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanodiff.sampler import generate as upstream_generate  # noqa: E402

from bijou.backends.nanodiff import NanoDiffBackend, tiny_config  # noqa: E402
from bijou.core.config import load  # noqa: E402
from bijou.core.types import GenerationRequest  # noqa: E402


class IdTokenizer:
    """One codepoint per token, so decode is the inverse of encode for every id."""

    eot_token = 50256

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _upstream(backend: NanoDiffBackend, req: GenerationRequest) -> str:
    """Upstream generate on the same request, decoded the way the backend decodes."""
    ids = backend.prompt_ids(req.prompt)
    prompt = torch.tensor([ids], device=backend.nano.device)
    x = upstream_generate(
        backend.model, prompt, req.gen_length, req.steps, req.block_length, temperature=0.0
    )
    out = x[0, len(ids) :].tolist()
    if backend.eot_id in out:
        out = out[: out.index(backend.eot_id)]
    return backend.enc.decode(out)


@pytest.mark.parametrize("steps,gen_length,block_length", [(4, 16, 8), (6, 16, 4), (8, 16, 16)])
def test_loop_matches_upstream_on_cpu(cfg, steps, gen_length, block_length):
    torch.manual_seed(0)
    backend = NanoDiffBackend(cfg, nano=tiny_config(), tokenizer=IdTokenizer())
    backend.build()
    req = GenerationRequest(
        prompt="instruction", gen_length=gen_length, steps=steps, block_length=block_length
    )
    assert backend.generate(req) == _upstream(backend, req)


@pytest.mark.gpu
def test_loop_matches_upstream_on_base_checkpoint():
    cfg = load()
    if not (cfg.paths.base_checkpoints / f"{cfg.backend.checkpoint}.pt").exists():
        pytest.skip(f"no base checkpoint {cfg.backend.checkpoint}")
    backend = NanoDiffBackend(cfg)
    backend.build()
    req = GenerationRequest(
        prompt="Extract name, role, city and years from the text as JSON.\n\n"
        "Ana has worked as an engineer in Tempe for 7 years.",
        gen_length=cfg.sampling.gen_length,
        steps=cfg.sampling.steps,
        block_length=cfg.sampling.block_length,
    )
    assert backend.generate(req) == _upstream(backend, req)
