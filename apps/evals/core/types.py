"""Cases, the part of a engine result the suites read, scores and reports.

RunView is evals' own copy of the engine RunResult, holding only the fields a suite reads. The
engine HTTP surface is the contract; evals never imports the engine.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvalsError(Exception):
    """A case file is malformed, the engine cannot be reached, or a report is missing."""


class Expect(BaseModel):
    """What a case expects. A suite with nothing to check for a case does not count it."""

    model_config = ConfigDict(extra="forbid")

    status: str | None = None
    min_steps: int | None = None
    max_steps: int | None = None
    # Each must be equipped by some step.
    skills: list[str] | None = None
    no_skills: bool = False
    # At least one must be called. A trailing * matches a prefix, as in browser_*.
    tools: list[str] | None = None
    no_tools: bool = False
    # Every one must appear in the answer, ignoring case.
    answer_contains: list[str] = Field(default_factory=list)
    max_latency_ms: int | None = None


class EvalCase(BaseModel):
    """One golden request."""

    model_config = ConfigDict(extra="forbid")

    id: str
    request: str
    suites: list[str]
    expect: Expect = Field(default_factory=Expect)


class StepView(BaseModel):
    id: str
    status: str
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class PlanView(BaseModel):
    fallback: bool = False
    steps: list[dict[str, object]] = Field(default_factory=list)


class RunView(BaseModel):
    """The fields of a engine RunResult the suites read."""

    session_id: str
    status: str
    answer: str
    plan: PlanView
    steps: list[StepView]
    duration_ms: int

    @property
    def skills(self) -> set[str]:
        return {s for step in self.steps for s in step.skills}

    @property
    def tools(self) -> set[str]:
        return {t for step in self.steps for t in step.tools}


class Score(BaseModel):
    passed: bool
    detail: str = ""


class CaseResult(BaseModel):
    case_id: str
    suite: str
    score: Score
    session_id: str


class SuiteReport(BaseModel):
    suite: str
    passed: int
    total: int

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class EvalReport(BaseModel):
    """One eval run. Written to evals.report_path; the baseline is promoted from one."""

    engine_url: str
    git_sha: str
    started_at: str
    cases: int
    results: list[CaseResult]
    suites: list[SuiteReport]
