"""One module per suite, each exposing score(case, run) -> Score | None.

None means the case carries no expectation for that suite, so it is not counted.

  status    how the run ended
  plan      how many steps the planner made, and that it did not fall back
  skills    which diffusion skills were equipped, or that none were
  tools     which tools were called, or that none were
  answer    what the answer says
  latency   how long the run took
"""

from __future__ import annotations

import importlib
from typing import Protocol, cast

from evals.core.types import EvalCase, EvalsError, RunView, Score

NAMES = ("status", "plan", "skills", "tools", "answer", "latency")


class Suite(Protocol):
    def score(self, case: EvalCase, run: RunView) -> Score | None: ...


def load(name: str) -> Suite:
    if name not in NAMES:
        raise EvalsError(f"unknown suite {name}; known: {', '.join(NAMES)}")
    return cast(Suite, importlib.import_module(f"evals.suites.{name}"))
