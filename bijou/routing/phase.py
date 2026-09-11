"""Trajectory-phase activation policies.

A masked diffusion LM does not emit left to right. The sampler runs rounds of
predict-everything then commit-the-confident. Early rounds settle coarse
structure, late rounds settle contested positions. Which adapter is live at
which point along that trajectory is an axis with no autoregressive counterpart,
and it is the project's only diffusion-specific claim.

    progress = 0.0 (all masked)  ------------->  1.0 (all committed)
      [ plan ]
              [ ---- domain ---- ]
                                  [ verify ]

Static application is expressed as one phase spanning the whole trajectory, so
the static and routed conditions run through the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass

from bijou.core.types import AdapterState, ConfigError


@dataclass(frozen=True)
class Phase:
    """Adapters live over the fractional interval [start, end)."""

    start: float
    end: float
    adapters: dict[str, float]

    def __post_init__(self) -> None:
        if not 0.0 <= self.start < self.end <= 1.0:
            raise ConfigError(f"phase interval [{self.start}, {self.end}) is not ordered")


class PhaseSchedule:
    """An ordered set of phases. Overlaps are allowed and additive."""

    def __init__(self, *phases: Phase) -> None:
        if not phases:
            raise ConfigError("a schedule needs at least one phase")
        self.phases = list(phases)

    @classmethod
    def static(cls, *names: str) -> PhaseSchedule:
        """The control condition: one set live for the whole trajectory."""
        return cls(Phase(0.0, 1.0, {n: 1.0 for n in names}))

    @classmethod
    def split(cls, early: str, late: str, at: float = 0.5) -> PhaseSchedule:
        """One adapter before the split point, another after it."""
        return cls(Phase(0.0, at, {early: 1.0}), Phase(at, 1.0, {late: 1.0}))

    def active_at(self, progress: float) -> dict[str, float]:
        merged: dict[str, float] = {}
        for phase in self.phases:
            if phase.start <= progress < phase.end:
                for name, weight in phase.adapters.items():
                    merged[name] = merged.get(name, 0.0) + weight
        return merged

    def validate_against_blocks(self, steps: int, blocks: int) -> None:
        """Reject a boundary that falls inside a sampler block.

        Generation with a prefix K/V cache prefills once per block and reuses it
        across that block's steps. Changing adapters mid-block leaves the cache
        describing weights that are no longer active.
        """
        if blocks <= 1:
            return
        per_block = steps / blocks
        for phase in self.phases:
            for edge in (phase.start, phase.end):
                position = edge * steps / per_block
                if abs(position - round(position)) > 1e-9:
                    raise ConfigError(
                        f"phase boundary at step {edge * steps:g} falls inside a "
                        f"block of {per_block:g} steps; align it or disable the cache"
                    )


class PhaseRouter:
    """Binds a schedule to a live AdapterState. Satisfies ActivationPolicy."""

    def __init__(self, state: AdapterState, schedule: PhaseSchedule) -> None:
        self.state = state
        self.schedule = schedule

    def active_at(self, progress: float) -> dict[str, float]:
        return self.schedule.active_at(progress)

    def at(self, step: int, total_steps: int) -> dict[str, float]:
        """Apply the schedule for one denoising step. Called by the sampler."""
        self.state.active = self.schedule.active_at(step / max(total_steps, 1))
        return self.state.active

    def __enter__(self) -> PhaseRouter:
        self._saved = dict(self.state.active)
        return self

    def __exit__(self, *exc: object) -> None:
        self.state.active = self._saved
