"""The subagent loop: answers, tools, repeats, limits, policy, confirmation and skills."""

import time

from engine.agent.policy import RiskPolicy
from engine.agent.subagent import Subagent
from engine.agent.toolset import ToolSet
from engine.core.doubles import (
    FakeSkillRuntime,
    FakeTool,
    MemoryTrace,
    ScriptedModel,
    calls,
    text,
)
from engine.core.types.agent import (
    ModelResponse,
    PhaseSpec,
    PlanStep,
    RequestContext,
    RiskClass,
    Role,
    RunStatus,
    SkillInfo,
    SkillPick,
    ToolCall,
    TraceKind,
)
from engine.core.types.errors import ModelError, ToolError

STEP = PlanStep(id="1", goal="do the thing")


def tuned(cfg, table, **values):
    return cfg.model_copy(update={table: getattr(cfg, table).model_copy(update=values)})


def make(cfg, replies, tools=(), runtime=None):
    trace = MemoryTrace()
    model = ScriptedModel(replies)
    runtime = runtime or FakeSkillRuntime()
    agent = Subagent(model, ToolSet(tools), runtime, RiskPolicy(cfg.policy), trace, cfg)
    return agent, model, trace


async def test_a_reply_without_tool_calls_answers(cfg, ctx):
    agent, _, trace = make(cfg, [text("done")])
    outcome = await agent.run(ctx, STEP, "the request", "", SkillPick())
    assert outcome.report.status is RunStatus.ANSWERED
    assert outcome.report.answer == "done"
    assert outcome.report.turns == 1
    assert trace.kinds()[-1] is TraceKind.STEP_DONE


async def test_the_opening_carries_the_goal_the_request_and_earlier_results(cfg, ctx):
    agent, model, _ = make(cfg, [text("done")])
    await agent.run(ctx, STEP, "the request", "step 0 found X", SkillPick())
    user = model.requests[0].messages[1].content
    assert "do the thing" in user and "the request" in user and "step 0 found X" in user


async def test_tool_results_are_fed_back_before_the_answer(cfg, ctx):
    tool = FakeTool("lookup")
    agent, model, trace = make(cfg, [calls(("lookup", {"q": "x"})), text("found it")], [tool])
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick())
    assert tool.calls == [{"q": "x"}]
    last = model.requests[1].messages[-1]
    assert last.role is Role.TOOL and "lookup says" in last.content
    assert outcome.report.tools == ["lookup"]
    assert TraceKind.TOOL_RESULT in trace.kinds()


async def test_a_repeated_call_is_refused_then_tools_are_withdrawn(cfg, ctx):
    tool = FakeTool("lookup")
    replies = [calls(("lookup", {"q": "x"})), calls(("lookup", {"q": "x"})), text("ok")]
    agent, model, _ = make(cfg, replies, [tool])
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick())
    assert len(tool.calls) == 1
    assert model.requests[2].tools == []
    assert outcome.report.status is RunStatus.ANSWERED


async def test_calling_tools_after_they_are_withdrawn_stalls(cfg, ctx):
    tool = FakeTool("lookup")
    replies = [
        calls(("lookup", {"q": "x"})),
        calls(("lookup", {"q": "x"})),
        calls(("lookup", {"q": "y"})),
        calls(("lookup", {"q": "z"})),
    ]
    agent, _, _ = make(cfg, replies, [tool])
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick())
    assert outcome.report.status is RunStatus.STALLED
    assert len(tool.calls) == 1


async def test_the_turn_limit_stops_the_step(cfg, ctx):
    tool = FakeTool("lookup")
    replies = [calls(("lookup", {"q": "a"})), calls(("lookup", {"q": "b"}))]
    agent, _, _ = make(tuned(cfg, "loop", max_turns=2), replies, [tool])
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick())
    assert outcome.report.status is RunStatus.STEP_LIMIT
    assert outcome.report.turns == 2


async def test_the_deadline_stops_the_step_before_any_call(cfg):
    agent, model, _ = make(cfg, [text("never")])
    outcome = await agent.run(RequestContext.start("s", 0), STEP, "r", "", SkillPick())
    assert outcome.report.status is RunStatus.DEADLINE
    assert model.requests == []


async def test_a_denied_tool_is_reported_to_the_model(cfg, ctx):
    tool = FakeTool("secret", risk=RiskClass.FORBIDDEN)
    agent, model, trace = make(cfg, [calls(("secret", {})), text("fine")], [tool])
    await agent.run(ctx, STEP, "r", "", SkillPick())
    assert tool.calls == []
    assert model.requests[1].messages[-1].content.startswith("denied")
    policy = [e for e in trace.events if e.kind is TraceKind.POLICY]
    assert policy[0].data["decision"] == "deny"


async def test_an_unknown_tool_and_bad_arguments_are_errors_the_model_reads(cfg, ctx):
    bad = ModelResponse(
        tool_calls=[
            ToolCall(id="c1", name="nope"),
            ToolCall(id="c2", name="lookup", error="Expecting value"),
        ]
    )
    agent, model, _ = make(cfg, [bad, text("ok")], [FakeTool("lookup")])
    await agent.run(ctx, STEP, "r", "", SkillPick())
    first, second = model.requests[1].messages[-2:]
    assert "no tool named nope" in first.content
    assert "not valid JSON" in second.content


async def test_a_failing_or_slow_tool_does_not_end_the_step(cfg, ctx):
    broken = FakeTool("broken", error=ToolError("boom"))
    slow = FakeTool("slow", delay=1.0)
    replies = [calls(("broken", {}), ("slow", {})), text("ok")]
    agent, model, _ = make(tuned(cfg, "loop", tool_timeout_secs=0.05), replies, [broken, slow])
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick())
    first, second = (m.content for m in model.requests[1].messages[-2:])
    assert first == "error: boom"
    assert "took longer" in second
    assert outcome.report.status is RunStatus.ANSWERED


async def test_long_tool_output_is_truncated(cfg, ctx):
    tool = FakeTool("big", answer=lambda _: "x" * 100)
    agent, model, _ = make(
        tuned(cfg, "loop", max_tool_output_chars=10), [calls(("big", {})), text("ok")], [tool]
    )
    await agent.run(ctx, STEP, "r", "", SkillPick())
    assert "[truncated, 90 more characters]" in model.requests[1].messages[-1].content


async def test_independent_calls_run_concurrently(cfg, ctx):
    tools = [FakeTool("a", delay=0.2), FakeTool("b", delay=0.2)]
    agent, _, _ = make(cfg, [calls(("a", {}), ("b", {})), text("ok")], tools)
    started = time.monotonic()
    await agent.run(ctx, STEP, "r", "", SkillPick())
    assert time.monotonic() - started < 0.35


async def test_a_sequential_tool_makes_the_turn_sequential(cfg, ctx):
    tools = [FakeTool("a", delay=0.15, sequential=True), FakeTool("b", delay=0.15)]
    agent, _, _ = make(cfg, [calls(("a", {}), ("b", {})), text("ok")], tools)
    started = time.monotonic()
    await agent.run(ctx, STEP, "r", "", SkillPick())
    assert time.monotonic() - started >= 0.3


async def test_a_write_waits_for_confirmation_and_resumes(cfg, ctx):
    post = FakeTool("post", risk=RiskClass.EXTERNAL_WRITE)
    agent, _, _ = make(cfg, [calls(("post", {"q": "hi"})), text("posted")], [post])
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick())
    assert outcome.report.status is RunStatus.AWAITING_CONFIRMATION
    assert outcome.pending is not None and post.calls == []
    assert "post" in outcome.pending.confirmation.summary
    resumed = await agent.resume(ctx, outcome.pending, approved=True)
    assert post.calls == [{"q": "hi"}]
    assert resumed.report.status is RunStatus.ANSWERED
    assert resumed.report.answer == "posted"
    assert resumed.report.tools == ["post"]


async def test_a_declined_action_is_not_run(cfg, ctx):
    post = FakeTool("post", risk=RiskClass.EXTERNAL_WRITE)
    agent, model, _ = make(cfg, [calls(("post", {})), text("skipped it")], [post])
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick())
    resumed = await agent.resume(ctx, outcome.pending, approved=False)
    assert post.calls == []
    assert "declined" in model.requests[-1].messages[-1].content
    assert resumed.report.answer == "skipped it"


async def test_only_the_first_of_two_writes_is_held(cfg, ctx):
    post = FakeTool("post", risk=RiskClass.EXTERNAL_WRITE)
    mail = FakeTool("mail", risk=RiskClass.EXTERNAL_WRITE)
    read = FakeTool("read")
    agent, _, _ = make(cfg, [calls(("post", {}), ("mail", {}), ("read", {}))], [post, mail, read])
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick())
    assert outcome.pending.call.name == "post"
    results = {m.name: m.content for m in outcome.pending.messages if m.role is Role.TOOL}
    assert "waiting for the user" in results["mail"]
    assert read.calls == [{}]


async def test_equipped_skills_add_run_skill_and_reach_the_runtime(cfg, ctx):
    runtime = FakeSkillRuntime([SkillInfo(name="json_extract", description="d", trained=True)])
    replies = [calls(("run_skill", {"input": "Ana is an engineer."})), text('{"name": "Ana"}')]
    agent, model, _ = make(cfg, replies, runtime=runtime)
    pick = SkillPick(skills=["json_extract"])
    outcome = await agent.run(ctx, STEP, "r", "", pick, {"json_extract": "pulls fields out"})
    assert "run_skill" in [t.name for t in model.requests[0].tools]
    assert "json_extract: pulls fields out" in model.requests[0].messages[0].content
    assert runtime.requests[0].skills == ["json_extract"]
    assert runtime.requests[0].prompt == "Ana is an engineer."
    assert outcome.report.skills == ["json_extract"]


async def test_a_schedule_is_sent_to_the_runtime_as_it_was_picked(cfg, ctx):
    runtime = FakeSkillRuntime()
    schedule = [
        PhaseSpec(start=0.0, end=0.5, skills={"plan": 1.0}),
        PhaseSpec(start=0.5, end=1.0, skills={"verify": 1.0}),
    ]
    agent, _, _ = make(cfg, [calls(("run_skill", {"input": "x"})), text("ok")], runtime=runtime)
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick(schedule=schedule))
    assert runtime.requests[0].schedule == schedule
    assert runtime.requests[0].skills == []
    assert outcome.report.skills == ["plan", "verify"]


async def test_run_skill_is_absent_without_equipped_skills(cfg, ctx):
    agent, model, _ = make(cfg, [text("ok")], [FakeTool("lookup")])
    await agent.run(ctx, STEP, "r", "", SkillPick())
    assert [t.name for t in model.requests[0].tools] == ["lookup"]


async def test_a_retryable_model_error_is_retried(cfg, ctx):
    agent, _, trace = make(cfg, [ModelError("down", retryable=True), text("ok")])
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick())
    assert outcome.report.status is RunStatus.ANSWERED
    errors = [e for e in trace.events if e.kind is TraceKind.MODEL_ERROR]
    assert errors[0].data["retried"] is True


async def test_a_permanent_model_error_ends_the_step(cfg, ctx):
    agent, _, _ = make(cfg, [ModelError("bad request")])
    outcome = await agent.run(ctx, STEP, "r", "", SkillPick())
    assert outcome.report.status is RunStatus.ERROR
    assert "bad request" in outcome.report.answer
