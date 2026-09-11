"""The composition matrix.

Rows are conditions, columns are skill evals. eval.conditions selects the rows:

  adapters       every subset of the trained adapters, the empty set being zero-shot
  tuned-prompt   the base model, each skill under the prompt tuned for it
  full-finetune  one row per skill, the base model fully fine-tuned on that skill

An adapter or fine-tune that wins its own column by damaging the others is not
modular, so those rows are scored against every skill.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

from engine.core.config import Config
from engine.core.runs import RunRecord, digest
from engine.core.types.diffusion import AdapterState, Sample, SkillReport
from engine.core.types.errors import ArtifactError
from engine.routing.phase import PhaseSchedule
from engine.runtime.evaluate import (
    BackendFactory,
    default_backend,
    prepare,
    score_condition,
)
from engine.runtime.prompting import tune, tuned_eval_split

if TYPE_CHECKING:
    from engine.backends.nanodiff import NanoDiffBackend


def conditions(skills: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Every subset of the trained adapters, smallest first, plus the empty set."""
    out: dict[str, tuple[str, ...]] = {"none": ()}
    for size in range(1, len(skills) + 1):
        for combo in combinations(skills, size):
            out["+".join(combo)] = combo
    return out


def _require(path: Path, fix: str) -> None:
    if not path.exists():
        raise ArtifactError(f"{path} is missing; run: {fix}")


def run(cfg: Config, make_backend: BackendFactory = default_backend) -> list[SkillReport]:
    """Score every configured condition against every skill and write one run record."""
    skills = cfg.eval.skills
    selected = cfg.eval.conditions
    record = RunRecord.start("eval", cfg.model_dump(mode="json"))
    record.notes = f"matrix over {len(skills)} skills: {', '.join(selected)}"

    adapters = list(skills) if "adapters" in selected else []
    for skill in adapters:
        _require(cfg.adapter_path(skill), f"just skill train {skill}")
    if "full-finetune" in selected:
        for skill in skills:
            _require(cfg.full_finetune_path(skill), f"just skill train {skill} --full-finetune")

    reports: list[SkillReport] = []

    def score(
        backend: NanoDiffBackend,
        state: AdapterState,
        condition: str,
        skill: str,
        schedule: PhaseSchedule,
        samples: list[Sample] | None = None,
    ) -> None:
        report, _ = score_condition(cfg, backend, state, skill, condition, schedule, samples)
        reports.append(report)
        record.scores[f"{condition}/{skill}"] = report.rate

    if "adapters" in selected or "tuned-prompt" in selected:
        backend, state = prepare(cfg, adapters, backend=make_backend(cfg))
        if backend.checkpoint_path is not None:
            record.inputs["base_checkpoint"] = digest(backend.checkpoint_path)
        for skill in adapters:
            record.inputs[f"adapter/{skill}"] = digest(cfg.adapter_path(skill))

        if "adapters" in selected:
            for name, active in conditions(skills).items():
                for skill in skills:
                    score(backend, state, name, skill, schedule=PhaseSchedule.static(*active))

        if "tuned-prompt" in selected:
            for skill in skills:
                choice = tune(cfg, backend, state, skill)
                record.scores[f"tuned-prompt/{skill}/instruction"] = float(choice.instruction)
                record.scores[f"tuned-prompt/{skill}/shots"] = float(choice.shots)
                record.scores[f"tuned-prompt/{skill}/dev_rate"] = choice.dev_rate
                score(
                    backend,
                    state,
                    "tuned-prompt",
                    skill,
                    schedule=PhaseSchedule.static(),
                    samples=tuned_eval_split(cfg, skill, choice),
                )

    if "full-finetune" in selected:
        for trained in skills:
            path = cfg.full_finetune_path(trained)
            full_cfg = cfg.model_copy(
                update={"backend": cfg.backend.model_copy(update={"checkpoint": path.stem})}
            )
            backend, state = prepare(full_cfg, [], backend=make_backend(full_cfg))
            record.inputs[f"full-finetune/{trained}"] = digest(path)
            for skill in skills:
                score(backend, state, f"full:{trained}", skill, schedule=PhaseSchedule.static())

    record.finish()
    record.write(cfg.paths.runs)
    return reports
