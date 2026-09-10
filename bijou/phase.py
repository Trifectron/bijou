"""Trajectory-phase routing: which adapter is live at which denoising step.

This is the only part of Bijou with no autoregressive analogue, and therefore
the part most worth testing first.

A masked diffusion LM does not emit left-to-right. nanoDiff's sampler runs
`steps` rounds of {predict every position, commit the most confident ones}.
Early rounds commit the tokens the model is surest about -- coarse structure.
Late rounds fill the residual, most-contested positions. If different learned
capabilities are useful at different points along that trajectory, then holding
one adapter fixed for the whole generation is leaving signal on the floor.

    t = 1.0  (all masked)  ------------------------->  t = 0  (answer)
      [ plan ]
              [ ---- domain ---- ]
                                  [ verify ]

Usage -- one line inside nanoDiff's `generate()` loop, immediately before each
`model(x)` / `model.forward_decode(...)` call:

    router.at(global_step, total_steps)

Everything else is the existing sampler. Because AdapterState.active is a plain
dict shared by reference with every LoRALinear, this costs a dict assignment.

CAVEAT: `generate(use_cache=True)` prefills K/V once per block and reuses it
across that block's steps. Changing adapters mid-block then leaves the cache
stale for the non-active positions. Phase boundaries must align with block
boundaries, or the cache must be off. `PhaseSchedule.validate_against_blocks`
checks this; the experiment runner should call it.

NOTE: not executed -- no torch in the authoring container.
"""
from __future__ import annotations

from dataclasses import dataclass

from .lora import AdapterState


@dataclass(frozen=True)
class Phase:
    """Adapters active over the fractional trajectory interval [start, end).

    `start`/`end` are fractions of total denoising steps, 0.0 = first step
    (fully masked) and 1.0 = last step (fully committed).
    """

    start: float
    end: float
    adapters: dict[str, float]

    def __post_init__(self):
        if not 0.0 <= self.start < self.end <= 1.0:
            raise ValueError(f"bad interval [{self.start}, {self.end})")


class PhaseSchedule:
    """An ordered set of phases. Overlaps are allowed and additive."""

    def __init__(self, *phases: Phase):
        self.phases = list(phases)

    @classmethod
    def static(cls, *names: str) -> "PhaseSchedule":
        """The control condition: one adapter set, live for the whole trajectory."""
        return cls(Phase(0.0, 1.0, {n: 1.0 for n in names}))

    def active_at(self, progress: float) -> dict[str, float]:
        merged: dict[str, float] = {}
        for phase in self.phases:
            if phase.start <= progress < phase.end:
                for name, weight in phase.adapters.items():
                    merged[name] = merged.get(name, 0.0) + weight
        return merged

    def validate_against_blocks(self, steps: int, n_blocks: int) -> None:
        """Fail loudly if a phase boundary falls inside a sampler block.

        See the K/V-cache caveat in the module docstring.
        """
        if n_blocks <= 1:
            return
        per_block = steps / n_blocks
        for phase in self.phases:
            for edge in (phase.start, phase.end):
                step = edge * steps
                if abs(step / per_block - round(step / per_block)) > 1e-9:
                    raise ValueError(
                        f"phase boundary at step {step:g} falls inside a sampler "
                        f"block ({per_block:g} steps each). Align the boundary or "
                        f"run generate(use_cache=False)."
                    )


class PhaseRouter:
    """Binds a schedule to a live AdapterState."""

    def __init__(self, state: AdapterState, schedule: PhaseSchedule):
        self.state = state
        self.schedule = schedule

    def at(self, step: int, total_steps: int) -> dict[str, float]:
        progress = step / max(total_steps, 1)
        self.state.active = self.schedule.active_at(progress)
        return self.state.active

    def __enter__(self):
        self._saved = dict(self.state.active)
        return self

    def __exit__(self, *exc):
        self.state.active = self._saved
        return False
