"""The live trace: what a run prints while it happens, one line per trace event."""

import io

from rich.console import Console

from engine.commands.live import LiveTrace, render
from engine.core.doubles import (
    FakeSkillRuntime,
    FakeTool,
    MemorySessionStore,
    RoutedModel,
    calls,
    text,
)
from engine.core.types.agent import SkillInfo, TraceEvent, TraceKind
from engine.wiring import build

SKILL = SkillInfo(name="json_extract", description="pulls fields out", trained=True)


def event(kind, **data):
    return TraceEvent(kind=kind, session_id="s", step_id=data.pop("step_id", ""), data=data)


def shown(*events):
    """Everything a LiveTrace prints for these events."""
    out = io.StringIO()
    sink = LiveTrace(Console(file=out, width=200, no_color=True, highlight=False))
    for one in events:
        sink.emit(one)
    return out.getvalue()


def test_the_plan_is_printed_with_every_step_and_its_dependencies():
    printed = shown(
        event(
            TraceKind.PLANNED,
            steps=[
                {"id": "1", "goal": "look up the weather", "depends_on": []},
                {"id": "2", "goal": "summarise it", "depends_on": ["1"]},
            ],
            fallback=False,
            reason="two things to do",
        )
    )
    assert "plan   2 steps · two things to do" in printed
    assert "1  look up the weather" in printed
    assert "2  summarise it · after 1" in printed


def test_a_fallback_plan_says_so():
    printed = shown(
        event(TraceKind.PLANNED, steps=[{"id": "1", "goal": "g"}], fallback=True, reason="no JSON")
    )
    assert "planning fell back · no JSON" in printed


def test_the_pick_names_the_skills_the_reason_and_how_long_choosing_took():
    printed = shown(
        event(
            TraceKind.SKILLS_PICKED,
            step_id="1",
            skills=["json_extract", "summarise"],
            schedule=[],
            reason="the step returns fields",
            duration_ms=420,
        )
    )
    assert "equip  [1] json_extract+summarise · chosen in 0.4s · the step returns fields" in printed


def test_a_schedule_is_printed_phase_by_phase():
    printed = shown(
        event(
            TraceKind.SKILLS_PICKED,
            step_id="1",
            skills=["a", "b"],
            schedule=[
                {"start": 0.0, "end": 0.5, "skills": {"a": 1.0}},
                {"start": 0.5, "end": 1.0, "skills": {"b": 1.0}},
            ],
            reason="",
            duration_ms=100,
        )
    )
    assert "0.00-0.50  a x1.0" in printed and "0.50-1.00  b x1.0" in printed


def test_an_equipped_skill_run_separates_equipping_from_generating():
    printed = shown(
        event(
            TraceKind.TOOL_RESULT,
            step_id="1",
            tool="run_skill",
            ok=True,
            chars=12,
            duration_ms=2400,
            skills=["json_extract"],
            generate_ms=900,
        )
    )
    assert (
        "skill  [1] run_skill · ok · json_extract · equipped in 1.5s · generated in 0.9s" in printed
    )


def test_a_pick_made_without_a_model_call_shows_no_time():
    printed = shown(
        event(
            TraceKind.SKILLS_PICKED,
            step_id="1",
            skills=[],
            schedule=[],
            reason="the skill server has no trained skills",
            duration_ms=0,
        )
    )
    assert "equip  [1] nothing equipped · the skill server has no trained skills" in printed


def test_model_calls_tools_steps_and_notices_each_get_a_line():
    printed = shown(
        event(
            TraceKind.MODEL_CALL,
            step_id="1",
            purpose="plan",
            duration_ms=1200,
            prompt_tokens=512,
            completion_tokens=87,
            tool_calls=["lookup"],
        ),
        event(TraceKind.TOOL_CALL, step_id="1", tool="lookup", arguments='{"q": "x"}'),
        event(TraceKind.TOOL_RESULT, step_id="1", tool="lookup", ok=True, chars=9, duration_ms=300),
        event(TraceKind.POLICY, step_id="1", tool="lookup", decision="allow", reason=""),
        event(TraceKind.STEP_DONE, step_id="1", status="answered", turns=2, skills=[], tools=[]),
        event(TraceKind.NOTICE, message="running without skills"),
    )
    assert "llm    [1] plan · 1.2s · 512+87 tokens · asked for lookup" in printed
    assert 'tool   [1] calling lookup · {"q": "x"}' in printed
    assert "tool   [1] lookup · ok · 0.3s · 9 chars" in printed
    assert "policy [1] lookup · allow" in printed
    assert "step   [1] answered · 2 turns" in printed
    assert "notice running without skills" in printed


def test_the_run_boundaries_print_nothing_of_their_own():
    assert render(event(TraceKind.RUN_STARTED, request="find x")) is None
    assert render(event(TraceKind.RUN_DONE, status="answered")) is None


async def test_a_whole_run_prints_the_plan_the_pick_and_the_tools_before_the_answer(cfg):
    out = io.StringIO()
    sink = LiveTrace(Console(file=out, width=200, no_color=True, highlight=False))
    worker = iter([calls(("lookup", {"q": "x"})), text("done")])
    agent = build(
        cfg,
        model=RoutedModel(
            {
                "You plan work": lambda req: text('{"steps": [{"id": "1", "goal": "look it up"}]}'),
                "You equip a worker": lambda req: text(
                    '{"skills": ["json_extract"], "reason": "fits"}'
                ),
                "You are one worker": lambda req: next(worker),
            }
        ),
        runtime=FakeSkillRuntime([SKILL]),
        sessions=MemorySessionStore(),
        tools=[FakeTool("lookup")],
        sinks=[sink],
    )
    await agent.orchestrator.run("find x")
    printed = out.getvalue()
    assert "plan   1 step" in printed
    assert "1  look it up" in printed
    assert "equip  [1] json_extract" in printed
    assert "tool   [1] lookup · ok" in printed
    assert "step   [1] answered" in printed
