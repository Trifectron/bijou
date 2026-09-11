"""One module per skill, each exposing generate(n, seed) and grade(sample, output).

A skill module also exposes INSTRUCTIONS, the candidate instructions for the
tuned-prompt baseline, and every sample it generates carries its input under
meta["text"], so a prompt can be rebuilt from an instruction and the input.

A skill is data and a grader. It imports no backend and no runtime, so graders
are unit-testable on CPU in milliseconds and the eval suite runs on every commit.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from bijou.core.types import BijouError

KNOWN = ("json_extract",)


class UnknownSkill(BijouError):
    """A skill name that is not in KNOWN."""


def load(name: str) -> ModuleType:
    """The module for one skill."""
    if name not in KNOWN:
        raise UnknownSkill(f"unknown skill {name}; known: {', '.join(KNOWN)}")
    return importlib.import_module(f"bijou.skills.{name}")
