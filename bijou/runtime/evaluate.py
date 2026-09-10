"""Scoring one condition against one skill's eval split.

A condition is a named activation policy: no adapter, one adapter, a sum of
adapters, or a phase schedule. Every condition runs the same loop, which is what
makes the composition matrix a comparison rather than a collection of scripts.
"""

from __future__ import annotations

from pathlib import Path

from bijou.adapters import io as adapter_io
from bijou.adapters.lora import inject
from bijou.backends.nanodiff import NanoDiffBackend
from bijou.core.config import Config
from bijou.core.determinism import seed_everything
from bijou.core.types import AdapterState, GenerationRequest, Score, SkillReport
from bijou.routing.phase import PhaseRouter, PhaseSchedule
from bijou.skills import load as load_skill


def prepare(cfg: Config, adapters: list[str]) -> tuple[NanoDiffBackend, AdapterState]:
    """Build a model with the named adapters attached but nothing active."""
    backend = NanoDiffBackend(cfg)
    model = backend.build()
    state = inject(model, cfg.adapter.targets)
    for name in adapters:
        adapter_io.load(model, Path(cfg.paths.adapters) / f"{name}.pt")
    return backend, state


def score_condition(
    cfg: Config,
    backend: NanoDiffBackend,
    state: AdapterState,
    skill_name: str,
    condition: str,
    schedule: PhaseSchedule,
) -> tuple[SkillReport, list[Score]]:
    """Generate and grade the eval split under one activation policy."""
    seed_everything(cfg.eval.seed)
    skill = load_skill(skill_name)
    samples = skill.generate(cfg.eval.eval_samples, cfg.eval.seed)

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
