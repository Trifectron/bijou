"""The interfaces every implementation satisfies.

Three of these are the project's seams. Everything else in the repo is
infrastructure for them.

  AdapterSite       how a delta attaches to one frozen base module
  ActivationPolicy  what is live at denoising step t
  Grader            how a skill's output is scored

Backend is the fourth interface, and exists so nanoDiff's internals are touched
in exactly one package.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bijou.core.types import (
    AdapterSpec,
    GenerationRequest,
    Sample,
    Score,
)

if TYPE_CHECKING:
    import torch


@runtime_checkable
class AdapterSite(Protocol):
    """A frozen module with a name-keyed bank of deltas attached to it.

    LoRA is one implementation. A trajectory-parameterized delta would be
    another, and must not require changes outside bijou.adapters.
    """

    def add(self, spec: AdapterSpec) -> None:
        """Attach a new delta. It must be a no-op until trained."""
        ...

    def names(self) -> Sequence[str]:
        """Every delta attached here."""
        ...

    def parameters_for(self, name: str) -> Iterator[torch.nn.Parameter]:
        """The trainable parameters of one delta."""
        ...


@runtime_checkable
class ActivationPolicy(Protocol):
    """Decides the active adapter set at a point along the denoising trajectory.

    Static application is the degenerate case: a policy that ignores progress.
    Holding every comparison behind one interface is what keeps the static and
    phase-routed conditions controlled against each other.
    """

    def active_at(self, progress: float) -> dict[str, float]:
        """Adapter name to weight, for progress in [0, 1). 0 is fully masked."""
        ...


@runtime_checkable
class Grader(Protocol):
    """Scores one generation against one sample. No model, no GPU, no network."""

    def __call__(self, sample: Sample, output: str) -> Score: ...


@runtime_checkable
class SkillData(Protocol):
    """Generates a skill's train and eval splits from a seed."""

    def generate(self, n: int, seed: int) -> list[Sample]: ...


@runtime_checkable
class Backend(Protocol):
    """A base model, its training step, and its sampler.

    The only package that may import a vendored model implementation.
    """

    def build(self) -> torch.nn.Module:
        """Construct the base model on the configured device."""
        ...

    def generate(self, req: GenerationRequest, on_step: object | None = None) -> str:
        """Run the reverse process. on_step is called before each denoising step."""
        ...
