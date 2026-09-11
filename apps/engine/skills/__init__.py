"""The skill registry: built-in skill modules and dataset skills collected onto disk.

A built-in skill is one module exposing NAME, DESCRIPTION, INSTRUCTIONS, PROMPT, generate and
grade. A dataset skill is a directory under paths.data/skills written by bijou collect, read by
DatasetSkill, and exposes the same surface. Both satisfy core.protocols.Skill, so training,
evaluation and the skill server treat them alike.

Every sample a skill generates carries its input under meta["text"], so a prompt can be rebuilt
from an instruction and the input.

A skill is data and a grader. It imports no backend, no runtime and no network client, so graders
are unit-testable on CPU in milliseconds and the eval suite runs on every commit.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import cast

from engine.core.protocols import Skill
from engine.core.types.errors import EngineError
from engine.skills.dataset import DatasetSkill, dataset_dir, dataset_names

KNOWN = ("json_extract",)


class UnknownSkill(EngineError):
    """A skill name that is neither built in nor collected under the data directory."""


def names(data: Path | None = None) -> tuple[str, ...]:
    """Every skill: the built-in ones, then the dataset skills under data, sorted."""
    collected = dataset_names(data) if data is not None else ()
    return KNOWN + tuple(n for n in collected if n not in KNOWN)


def load(name: str, data: Path | None = None) -> Skill:
    """One skill by name. data is paths.data, where collected dataset skills live."""
    if name in KNOWN:
        return cast(Skill, importlib.import_module(f"engine.skills.{name}"))
    if data is not None and name in dataset_names(data):
        return DatasetSkill.read(dataset_dir(data, name))
    raise UnknownSkill(f"unknown skill {name}; known: {', '.join(names(data))}")
