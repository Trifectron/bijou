"""Scoring one condition against one skill's eval split.

A condition is a named activation policy: no adapter, one adapter, a sum of
adapters, or a phase schedule. Every condition runs the same loop, which is what
makes the composition matrix a comparison rather than a collection of scripts.
"""

from __future__ import annotations

from collections.abc import Callable

from engine.adapters import io as adapter_io
from engine.adapters.lora import inject
from engine.backends.nanodiff import NanoDiffBackend
from engine.core.config import Config
from engine.core.determinism import seed_everything
from engine.core.types.diffusion import AdapterState, GenerationRequest, Sample, Score, SkillReport
from engine.routing.phase import PhaseRouter, PhaseSchedule
from engine.skills import load as load_skill

BackendFactory = Callable[[Config], NanoDiffBackend]


def default_backend(cfg: Config) -> NanoDiffBackend:
    """The configured backend."""
    return NanoDiffBackend(cfg)


def prepare(
    cfg: Config, adapters: list[str], backend: NanoDiffBackend | None = None
) -> tuple[NanoDiffBackend, AdapterState]:
    """Build a model with the named adapters attached but nothing active."""
    backend = backend or NanoDiffBackend(cfg)
    model = backend.build()
    state = inject(model, cfg.adapter.targets)
    for name in adapters:
        adapter_io.load(model, cfg.adapter_path(name))
    return backend, state


def score_condition(
    cfg: Config,
    backend: NanoDiffBackend,
    state: AdapterState,
    skill_name: str,
    condition: str,
    schedule: PhaseSchedule,
    samples: list[Sample] | None = None,
) -> tuple[SkillReport, list[Score]]:
    """Generate and grade the eval split, or the given samples, under one activation policy."""
    seed_everything(cfg.eval.seed)
    skill = load_skill(skill_name, cfg.paths.data)
    if samples is None:
        samples = skill.generate(cfg.eval.eval_samples, cfg.eval.seed, split="eval")

    blocks = cfg.sampling.gen_length // cfg.sampling.block_length
    schedule.validate_against_blocks(cfg.sampling.steps, blocks)
    router = PhaseRouter(state, schedule)

    scores = []
    for sample in samples:
        request = GenerationRequest(
            prompt=sample.prompt,
            gen_length=cfg.sampling.gen_length,
            steps=cfg.sampling.steps,
            block_length=cfg.sampling.block_length,
            temperature=cfg.sampling.temperature,
        )
        with router:
            output = backend.generate(request, on_step=router.at)
        scores.append(skill.grade(sample, output))

    report = SkillReport(
        skill=skill_name,
        condition=condition,
        passed=sum(s.passed for s in scores),
        total=len(scores),
        mean_value=sum(s.value for s in scores) / max(len(scores), 1),
    )
    return report, scores
