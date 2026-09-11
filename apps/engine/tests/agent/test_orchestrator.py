"""Whole requests: planning, skill selection, waves, synthesis, confirmation and resuming."""

import json

import pytest

from engine.agent.llm import extract_json
from engine.agent.planner import PlanReply, check_plan
from engine.agent.selector import PickReply, check_pick
from engine.core.doubles import (
    FakeSkillRuntime,
    FakeTool,
    MemorySessionStore,
    RoutedModel,
    calls,
    text,
)
from engine.core.types.agent import (
    ModelResponse,
    PhaseSpec,
    PlanStep,
    RiskClass,
    RunStatus,
    SkillInfo,
    TraceKind,
)
from engine.core.types.errors import ConfirmationError, PlanError, SkillRuntimeError, StoreError
from engine.wiring import build

JSON_SKILL = SkillInfo(name="json_extract", description="pulls fields out as JSON", trained=True)


def plan_of(*steps):
    return lambda req: text(json.dumps({"steps": list(steps), "reason": "because"}))


def worker(req):
    """Answers with the step goal, so tests can see which step produced what."""
    goal = req.messages[1].content.split("\n")[0].removeprefix("Your step: ")
    return text(f"did {goal}")


def harness_for(cfg, routes, tools=(), skills=(), runtime=None, sessions=None):
    model = RoutedModel(routes)
    runtime = runtime or FakeSkillRuntime(list(skills))
    sessions = sessions or MemorySessionStore()
    return build(cfg, model=model, runtime=runtime, sessions=sessions, tools=list(tools)), model


BASE = {
    "You plan work": plan_of({"id": "1", "goal": "only step"}),
    "You equip a worker": lambda req: text('{"skills": [], "reason": "none fit"}'),
    "You are one worker": worker,
    "Several workers": lambda req: text("combined answer"),
}


# ---------- pure checks ----------


def test_extract_json_ignores_reasoning_and_fences():
    assert extract_json('<think>{"no": 1}</think>```json\n{"a": 1}\n```') == '{"a": 1}'
    with pytest.raises(PlanError):
        extract_json("no json here")


def test_a_plan_may_only_depend_on_earlier_steps():
    good = PlanReply(
        steps=[PlanStep(id="a", goal="x"), PlanStep(id="b", goal="y", depends_on=["a"])]
    )
    check_plan(good, 4)
    forward = PlanReply(
        steps=[PlanStep(id="a", goal="x", depends_on=["b"]), PlanStep(id="b", goal="y")]
    )
    with pytest.raises(PlanError, match="not an earlier step"):
        check_plan(forward, 4)
    with pytest.raises(PlanError, match="at most 1"):
        check_plan(good, 1)
    with pytest.raises(PlanError, match="twice"):
        check_plan(PlanReply(steps=[PlanStep(id="a", goal="x"), PlanStep(id="a", goal="y")]), 4)


def test_a_pick_is_checked_against_the_trained_skills():
    trained = {"a", "b", "c"}
    check_pick(PickReply(skills=["a"]), trained, 2, True)
    with pytest.raises(PlanError, match="cannot equip"):
        check_pick(PickReply(skills=["z"]), trained, 2, True)
    with pytest.raises(PlanError, match="at most 2"):
        check_pick(PickReply(skills=["a", "b", "c"]), trained, 2, True)
    schedule = [
        PhaseSpec(start=0.0, end=0.5, skills={"a": 1}),
        PhaseSpec(start=0.5, end=1.0, skills={"b": 1}),
    ]
    check_pick(PickReply(schedule=schedule), trained, 2, True)
    with pytest.raises(PlanError, match="not allowed"):
        check_pick(PickReply(schedule=schedule), trained, 2, False)
    with pytest.raises(PlanError, match="inside"):
        check_pick(
            PickReply(schedule=[PhaseSpec(start=0.6, end=0.2, skills={"a": 1})]), trained, 2, True
        )


# ---------- whole runs ----------


async def test_a_one_step_request_answers_with_that_step(cfg):
    harness, model = harness_for(cfg, BASE)
    result = await harness.orchestrator.run("say hi")
    assert result.status is RunStatus.ANSWERED
    assert result.answer == "did only step"
    assert not any("Several workers" in r.messages[0].content for r in model.requests)
    assert harness.sessions.get(result.session_id).answer == "did only step"
    assert [e.kind for e in result.events][0] is TraceKind.RUN_STARTED


async def test_steps_run_in_dependency_waves_and_results_flow_forward(cfg):
    routes = BASE | {
        "You plan work": plan_of(
            {"id": "a", "goal": "find jobs"},
            {"id": "b", "goal": "find salaries"},
            {"id": "c", "goal": "compare", "depends_on": ["a", "b"]},
        )
    }
    harness, model = harness_for(cfg, routes)
    result = await harness.orchestrator.run("amazon scientist jobs")
    assert [s.id for s in result.steps] == ["a", "b", "c"]
    compare = next(
        r for r in model.requests if r.messages[1].content.startswith("Your step: compare")
    )
    assert "did find jobs" in compare.messages[1].content
    assert "did find salaries" in compare.messages[1].content
    assert result.answer == "combined answer"


async def test_a_plan_the_model_cannot_produce_falls_back_to_one_step(cfg):
    routes = BASE | {"You plan work": lambda req: text("I refuse to write JSON")}
    harness, _ = harness_for(cfg, routes)
    result = await harness.orchestrator.run("do it")
    assert result.plan.fallback
    assert result.plan.steps[0].goal == "do it"
    assert result.status is RunStatus.ANSWERED


async def test_the_selector_equips_a_trained_skill_and_the_step_can_use_it(cfg):
    routes = BASE | {
        "You equip a worker": lambda req: text('{"skills": ["json_extract"], "reason": "fields"}'),
    }
    turns = iter([calls(("run_skill", {"input": "Ana, engineer, Tempe"})), text("extracted")])
    routes["You are one worker"] = lambda req: next(turns)
    runtime = FakeSkillRuntime([JSON_SKILL])
    harness, _ = harness_for(cfg, routes, runtime=runtime)
    result = await harness.orchestrator.run("pull fields from this bio")
    step = result.steps[0]
    assert step.skills == ["json_extract"] and step.tools == ["run_skill"]
    assert runtime.requests[0].skills == ["json_extract"]
    picked = [e for e in result.events if e.kind is TraceKind.SKILLS_PICKED]
    assert picked[0].data["reason"] == "fields"


async def test_an_untrained_pick_is_corrected_once_then_dropped(cfg):
    replies = iter(['{"skills": ["made_up"]}', '{"skills": ["still_made_up"]}'])
    routes = BASE | {"You equip a worker": lambda req: text(next(replies))}
    harness, _ = harness_for(cfg, routes, skills=[JSON_SKILL])
    result = await harness.orchestrator.run("x")
    assert result.steps[0].skills == []
    assert "selector failed" in result.steps[0].pick_reason


async def test_with_no_trained_skills_the_selector_makes_no_call(cfg):
    harness, model = harness_for(cfg, BASE)
    await harness.orchestrator.run("x")
    assert not any("You equip a worker" in r.messages[0].content for r in model.requests)


async def test_an_unreachable_skill_server_is_a_notice_not_a_failure(cfg):
    runtime = FakeSkillRuntime(error=SkillRuntimeError("down"))
    harness, _ = harness_for(cfg, BASE, runtime=runtime)
    result = await harness.orchestrator.run("x")
    assert result.status is RunStatus.ANSWERED
    notices = [e for e in result.events if e.kind is TraceKind.NOTICE]
    assert "running without skills" in notices[0].data["message"]


async def test_a_write_holds_the_run_until_confirmed(cfg):
    post = FakeTool("post", risk=RiskClass.EXTERNAL_WRITE)
    turns = iter([calls(("post", {"q": "hello"})), text("posted it")])
    routes = BASE | {"You are one worker": lambda req: next(turns)}
    harness, _ = harness_for(cfg, routes, tools=[post])
    held = await harness.orchestrator.run("post hello")
    assert held.status is RunStatus.AWAITING_CONFIRMATION
    assert held.confirmation is not None and post.calls == []
    with pytest.raises(ConfirmationError, match="no action waiting"):
        await harness.orchestrator.confirm(held.session_id, "wrong-token", True)
    done = await harness.orchestrator.confirm(held.session_id, held.confirmation.token, True)
    assert done.status is RunStatus.ANSWERED and done.answer == "posted it"
    assert post.calls == [{"q": "hello"}]
    with pytest.raises(ConfirmationError):
        await harness.orchestrator.confirm(held.session_id, held.confirmation.token, True)


async def test_an_expired_confirmation_is_refused(cfg):
    post = FakeTool("post", risk=RiskClass.EXTERNAL_WRITE)
    turns = iter([calls(("post", {}))])
    routes = BASE | {"You are one worker": lambda req: next(turns)}
    harness, _ = harness_for(cfg, routes, tools=[post])
    held = await harness.orchestrator.run("post")
    record = harness.sessions.get(held.session_id)
    stale = record.pending[0].confirmation
    record.pending[0].confirmation = stale.model_copy(
        update={"expires_at": stale.expires_at.replace(year=2000)}
    )
    harness.sessions.save(record)
    with pytest.raises(ConfirmationError, match="expired"):
        await harness.orchestrator.confirm(held.session_id, stale.token, True)


async def test_a_resumed_session_gives_the_planner_its_earlier_answer(cfg):
    harness, model = harness_for(cfg, BASE)
    first = await harness.orchestrator.run("track amazon jobs")
    await harness.orchestrator.run("anything new?", resume_from=first.session_id)
    planner_calls = [r for r in model.requests if "You plan work" in r.messages[0].content]
    assert "did only step" in planner_calls[-1].messages[1].content
    with pytest.raises(StoreError):
        await harness.orchestrator.run("x", resume_from="missing")


async def test_a_failed_synthesis_still_returns_every_step_result(cfg):
    routes = BASE | {
        "You plan work": plan_of({"id": "a", "goal": "one"}, {"id": "b", "goal": "two"}),
        "Several workers": lambda req: ModelResponse(content=""),
    }
    harness, _ = harness_for(cfg, routes)
    result = await harness.orchestrator.run("x")
    assert "did one" in result.answer and "did two" in result.answer


async def test_disabled_tools_are_removed(cfg):
    tuned = cfg.model_copy(update={"tools": cfg.tools.model_copy(update={"disabled": ["lookup"]})})
    harness, model = harness_for(tuned, BASE, tools=[FakeTool("lookup")])
    await harness.orchestrator.run("x")
    worker_request = next(
        r for r in model.requests if "You are one worker" in r.messages[0].content
    )
    assert worker_request.tools == []
