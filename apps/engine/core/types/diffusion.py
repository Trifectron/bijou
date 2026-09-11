"""Data shared across adapters, routing, skills, backends and runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Phase = Literal["train", "eval"]

# Which split of a skill's data a caller wants. A synthetic skill separates splits by seed and
# ignores this; a dataset skill reads a different file for each.
Split = Literal["train", "dev", "eval"]


@dataclass(frozen=True)
class AdapterSpec:
    """The shape of one adapter. Identifies a delta independently of its weights."""

    name: str
    rank: int
    alpha: float
    targets: tuple[str, ...]
    dropout: float = 0.0

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass
class AdapterState:
    """Which adapters are live, and at what weight.

    Held by reference by every injection site in a model. Mutating active changes
    the model's behaviour with no module traversal, which is what makes
    per-denoising-step routing affordable.
    """

    active: dict[str, float] = field(default_factory=dict)

    def set(self, *names: str, **weighted: float) -> None:
        self.active = {n: 1.0 for n in names} | dict(weighted)

    def clear(self) -> None:
        self.active = {}


@dataclass(frozen=True)
class Sample:
    """One skill example. prompt conditions the model, target is the graded answer."""

    id: str
    prompt: str
    target: str
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Score:
    """One graded generation. detail explains a failure and is empty on a pass."""

    passed: bool
    value: float
    detail: str = ""


@dataclass(frozen=True)
class SkillReport:
    """Aggregate of one skill's eval under one condition."""

    skill: str
    condition: str
    passed: int
    total: int
    mean_value: float

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(frozen=True)
class GenerationRequest:
    """What runtime asks a backend to generate. Backend-agnostic on purpose."""

    prompt: str
    gen_length: int
    steps: int
    block_length: int | None = None
    temperature: float = 0.0
