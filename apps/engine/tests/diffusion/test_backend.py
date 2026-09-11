"""The backend: encoding, the SFT objective, and the denoising loop.

A stub tokenizer stands in for tiktoken, whose vocabulary is fetched over the
network on first use. These tests exercise the reverse process and the
ActivationPolicy hook on CPU, so the loop is covered without a GPU or a download.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine.adapters.lora import add, inject  # noqa: E402
from engine.backends.nanodiff import NanoDiffBackend, tiny_config  # noqa: E402
from engine.core.types.diffusion import AdapterSpec, GenerationRequest  # noqa: E402
from engine.core.types.errors import BackendError  # noqa: E402
from engine.routing.phase import PhaseRouter, PhaseSchedule  # noqa: E402


@pytest.fixture
def backend(cfg, make_backend):
    return make_backend(cfg)


def _with_checkpoint(cfg, directory, name):
    return cfg.model_copy(
        update={
            "backend": cfg.backend.model_copy(update={"checkpoint": name}),
            "paths": cfg.paths.model_copy(update={"base_checkpoints": directory}),
        }
    )


def test_a_missing_base_checkpoint_is_refused(cfg, make_backend, tmp_path):
    backend = make_backend(_with_checkpoint(cfg, tmp_path, "absent"))
    with pytest.raises(BackendError, match="just checkpoints"):
        backend.build()


def test_the_checkpoint_sets_the_architecture_and_weights(cfg, tmp_path):
    from nanodiff.model import NanoDiff

    torch.manual_seed(0)
    saved = NanoDiff(tiny_config(n_layer=3))
    torch.save({"model": saved.state_dict(), "config": tiny_config(n_layer=3)}, tmp_path / "t.pt")

    backend = NanoDiffBackend(_with_checkpoint(cfg, tmp_path, "t"))
    model = backend.build()

    assert backend.nano.n_layer == 3
    assert backend.nano.device == cfg.backend.device
    loaded = model.state_dict()
    assert all(torch.equal(v, loaded[k]) for k, v in saved.state_dict().items())


def test_generating_before_build_is_refused(backend):
    with pytest.raises(BackendError, match="build the model"):
        backend.generate(GenerationRequest(prompt="x", gen_length=8, steps=2))


def test_encoding_is_fixed_width(backend, cfg):
    prompt, response = backend.encode("instruction text", "the answer")
    assert prompt.shape == (cfg.train.prompt_len,)
    assert response.shape == (cfg.train.response_len,)


def test_sft_loss_is_finite(backend):
    backend.build()
    prompt, response = backend.encode("instruction", "answer")
    loss = backend.loss(prompt.unsqueeze(0), response.unsqueeze(0))
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_generation_is_bounded_by_gen_length(backend, cfg):
    backend.build()
    request = GenerationRequest(
        prompt="instruction",
        gen_length=cfg.sampling.gen_length,
        steps=cfg.sampling.steps,
        block_length=cfg.sampling.block_length,
    )
    out = backend.generate(request)
    # The stub decodes one character per token, and the schedule commits every
    # position, so the output is gen_length characters unless an eot truncates it.
    assert 0 < len(out) <= cfg.sampling.gen_length


def test_step_hook_fires_once_per_step(backend, cfg):
    backend.build()
    seen: list[tuple[int, int]] = []
    request = GenerationRequest(
        prompt="instruction",
        gen_length=cfg.sampling.gen_length,
        steps=cfg.sampling.steps,
        block_length=cfg.sampling.block_length,
    )
    backend.generate(request, on_step=lambda step, total: seen.append((step, total)))
    assert [s for s, _ in seen] == list(range(cfg.sampling.steps))
    assert all(total == cfg.sampling.steps for _, total in seen)


def test_phase_router_switches_adapters_during_generation(backend, cfg):
    model = backend.build()
    state = inject(model, cfg.adapter.targets)
    for name in ("early", "late"):
        add(model, AdapterSpec(name=name, rank=2, alpha=2.0, targets=cfg.adapter.targets))

    schedule = PhaseSchedule.split("early", "late", at=0.5)
    schedule.validate_against_blocks(
        cfg.sampling.steps, cfg.sampling.gen_length // cfg.sampling.block_length
    )
    router = PhaseRouter(state, schedule)

    observed: list[tuple[str, ...]] = []

    def hook(step: int, total: int) -> None:
        router.at(step, total)
        observed.append(tuple(state.active))

    request = GenerationRequest(
        prompt="instruction",
        gen_length=cfg.sampling.gen_length,
        steps=cfg.sampling.steps,
        block_length=cfg.sampling.block_length,
    )
    backend.generate(request, on_step=hook)

    assert observed[0] == ("early",)
    assert observed[-1] == ("late",)


def test_inert_adapter_does_not_change_generation(backend, cfg):
    request = GenerationRequest(
        prompt="instruction",
        gen_length=cfg.sampling.gen_length,
        steps=cfg.sampling.steps,
        block_length=cfg.sampling.block_length,
    )
    torch.manual_seed(0)
    model = backend.build()
    before = backend.generate(request)

    state = inject(model, cfg.adapter.targets)
    add(model, AdapterSpec(name="probe", rank=2, alpha=2.0, targets=cfg.adapter.targets))
    state.set("probe")
    after = backend.generate(request)

    assert before == after


def test_prompts_use_the_sft_template(cfg, make_backend):
    from nanodiff.sft import SFT_PROMPT_NO_INPUT

    roomy = cfg.model_copy(update={"train": cfg.train.model_copy(update={"prompt_len": 256})})
    backend = make_backend(roomy)
    ids = backend.prompt_ids("an instruction")
    assert backend.enc.decode(ids) == SFT_PROMPT_NO_INPUT.format(instruction="an instruction")


def test_a_long_prompt_keeps_the_response_cue(backend, cfg):
    ids = backend.prompt_ids("word " * 500)
    assert len(ids) == cfg.train.prompt_len
    assert backend.enc.decode(ids).endswith("### Response:\n")


def test_generation_restores_training_mode(backend, cfg):
    model = backend.build()
    model.train()
    backend.generate(
        GenerationRequest(prompt="x", gen_length=cfg.sampling.gen_length, steps=cfg.sampling.steps)
    )
    assert model.training
