"""Skill specs: what a skill should do, before any data for it exists.

A spec is a JSON file under paths.data/proposals. The harness pattern miner writes one when it
sees the same kind of work recur with no skill equipped; a person can also write one with
bijou collect new. Either way approved starts false, and collection refuses a spec until a
person has read it and set approved to true.

  name          lower snake case; becomes the adapter file and the skill name
  description   what the harness skill selector reads when deciding to equip it
  instruction   the instruction every example is prompted with
  examples      inputs seen in practice, which the teacher writes more of
  pairs         worked input and output pairs, which the teacher imitates and which are kept
  approved      set by a person, never by a program
  source        who proposed it
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from engine.core.types.errors import EngineError
from engine.skills.dataset import valid_name


class SpecError(EngineError):
    """A spec is malformed, unapproved, or names a skill that already exists."""


class Pair(BaseModel):
    """One worked example."""

    model_config = ConfigDict(extra="forbid")

    input: str
    output: str


class SkillSpec(BaseModel):
    """One proposed skill."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)
    pairs: list[Pair] = Field(default_factory=list)
    approved: bool = False
    source: str = "manual"
    occurrences: int = 0
    sessions: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not valid_name(value):
            raise ValueError("name must be lower snake case, 2 to 64 characters")
        return value


def read_spec(path: Path) -> SkillSpec:
    """Parse one spec file."""
    if not path.exists():
        raise SpecError(f"no spec at {path}")
    try:
        return SkillSpec.model_validate_json(path.read_text())
    except ValidationError as exc:
        raise SpecError(f"{path} is not a valid skill spec: {exc}") from exc


def write_spec(spec: SkillSpec, path: Path) -> None:
    """Write one spec file, refusing to replace one."""
    if path.exists():
        raise SpecError(f"{path} exists; edit it instead")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.model_dump(), indent=2) + "\n")


def list_specs(directory: Path) -> list[tuple[Path, SkillSpec | None]]:
    """Every spec file in a directory, with None for one that does not parse."""
    found: list[tuple[Path, SkillSpec | None]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            found.append((path, read_spec(path)))
        except SpecError:
            found.append((path, None))
    return found
