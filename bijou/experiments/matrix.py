"""The composition matrix.

Rows are activation conditions, columns are skill evals. An adapter that wins
its own column by damaging the others is not modular, so every condition is
scored against every skill.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from bijou.core.config import Config
from bijou.core.runs import RunRecord
from bijou.core.types import SkillReport
from bijou.routing.phase import PhaseSchedule
from bijou.runtime.evaluate import prepare, score_condition

if TYPE_CHECKING:
    from bijou.backends.nanodiff import NanoDiffBackend


def conditions(skills: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Every subset of the trained adapters, smallest first, plus the empty set."""
    out: dict[str, tuple[str, ...]] = {"none": ()}
    for size in range(1, len(skills) + 1):
        for combo in combinations(skills, size):
            out["+".join(combo)] = combo
    return out


def run(cfg: Config, backend: NanoDiffBackend | None = None) -> list[SkillReport]:
    """Score every condition against every skill and write one run record."""
    skills = cfg.eval.skills
    record = RunRecord.start("eval", cfg.model_dump(mode="json"))
    record.notes = f"matrix over {len(skills)} skills"

    backend, state = prepare(cfg, list(skills), backend=backend)
    reports = []
    for name, active in conditions(skills).items():
        schedule = PhaseSchedule.static(*active) if active else PhaseSchedule.static()
        for skill in skills:
            report, _ = score_condition(cfg, backend, state, skill, name, schedule)
            reports.append(report)
            record.scores[f"{name}/{skill}"] = report.rate

    record.finish()
    record.write(cfg.paths.runs)
    return reports
