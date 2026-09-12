"""Telemetry: metrics counted from trace events, a span tree for Phoenix, both off the loop."""

import urllib.request

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from engine.core.config import Telemetry
from engine.core.doubles import (
    FakeSkillRuntime,
    FakeTool,
    MemorySessionStore,
    RoutedModel,
    calls,
    text,
)
from engine.core.types.agent import RiskClass, SkillInfo, TraceEvent, TraceKind
from engine.telemetry.metrics import AgentMetrics, MetricsSink, new_registry, serve_metrics
from engine.telemetry.otel import OtelSink, open_tracer
from engine.wiring import build

SKILL = SkillInfo(name="json_extract", description="pulls fields out", trained=True)


def routes(*turns):
    worker = iter(turns)
    return {
        "You plan work": lambda req: text('{"steps": [{"id": "1", "goal": "look it up"}]}'),
        "You equip a worker": lambda req: text('{"skills": ["json_extract"], "reason": "fits"}'),
        "You are one worker": lambda req: next(worker),
    }


def agent_for(cfg, turns, tools=(), **kwargs):
    return build(
        cfg,
        model=RoutedModel(routes(*turns)),
        runtime=FakeSkillRuntime([SKILL]),
        sessions=MemorySessionStore(),
        tools=list(tools),
        **kwargs,
    )


def sample(registry, name, **labels):
    return registry.get_sample_value(name, labels)


async def test_a_run_is_counted_by_status_purpose_tool_and_pick(cfg):
    registry = new_registry()
    agent = agent_for(
        cfg,
        [calls(("lookup", {"q": "x"})), text("done")],
        [FakeTool("lookup")],
        registry=registry,
    )
    await agent.orchestrator.run("find x")
    assert sample(registry, "bijou_agent_runs_total", status="answered") == 1
    assert sample(registry, "bijou_agent_run_seconds_count") == 1
    assert sample(registry, "bijou_agent_steps_total", status="answered") == 1
    assert sample(registry, "bijou_agent_model_calls_total", purpose="plan", outcome="ok") == 1
    assert sample(registry, "bijou_agent_model_calls_total", purpose="select", outcome="ok") == 1
    assert sample(registry, "bijou_agent_model_calls_total", purpose="subagent", outcome="ok") == 2
    assert sample(registry, "bijou_agent_tokens_total", kind="prompt") == 40
    assert sample(registry, "bijou_agent_tool_calls_total", tool="lookup", ok="true") == 1
    assert sample(registry, "bijou_agent_tool_seconds_count", tool="lookup") == 1
    assert sample(registry, "bijou_agent_policy_decisions_total", decision="allow") == 1
    assert sample(registry, "bijou_agent_skill_picks_total", skills="json_extract") == 1


def test_a_malformed_event_is_not_counted_and_does_not_raise():
    registry = new_registry()
    MetricsSink(AgentMetrics(registry)).emit(
        TraceEvent(kind=TraceKind.RUN_DONE, session_id="s", data={})
    )
    assert sample(registry, "bijou_agent_run_seconds_count") == 0


async def test_model_calls_keep_the_prompt_and_reply_up_to_the_limit(cfg):
    agent = agent_for(cfg, [text("done")])
    result = await agent.orchestrator.run("find x")
    plan = next(e for e in result.events if e.data.get("purpose") == "plan")
    assert "You plan work" in plan.data["input"] and "look it up" in plan.data["output"]

    quiet = cfg.model_copy(update={"loop": cfg.loop.model_copy(update={"record_content_chars": 0})})
    result = await agent_for(quiet, [text("done")]).orchestrator.run("find x")
    assert all("input" not in e.data for e in result.events)


def spans_of(exporter):
    return exporter.get_finished_spans()


def tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OtelSink(provider.get_tracer("test")), exporter


async def test_a_run_becomes_a_span_tree_phoenix_can_read(cfg):
    sink, exporter = tracer()
    agent = agent_for(
        cfg, [calls(("lookup", {"q": "x"})), text("done")], [FakeTool("lookup")], sinks=[sink]
    )
    result = await agent.orchestrator.run("find x")
    spans = spans_of(exporter)
    names = sorted(s.name for s in spans)
    assert names == ["agent.run", "llm", "llm", "llm", "llm", "step", "tool"]

    root = next(s for s in spans if s.name == "agent.run")
    step = next(s for s in spans if s.name == "step")
    tool = next(s for s in spans if s.name == "tool")
    plan = next(s for s in spans if s.attributes.get("bijou.purpose") == "plan")
    assert root.attributes["session.id"] == result.session_id
    assert root.attributes["openinference.span.kind"] == "CHAIN"
    assert root.attributes["bijou.status"] == "answered"
    assert root.attributes["output.value"] == "done"
    assert step.parent.span_id == root.context.span_id
    assert plan.parent.span_id == root.context.span_id
    assert tool.parent.span_id == step.context.span_id
    assert step.attributes["bijou.skills"] == '["json_extract"]'
    assert tool.attributes["tool.name"] == "lookup" and '"x"' in tool.attributes["input.value"]
    assert plan.attributes["llm.token_count.prompt"] == 10
    assert (
        plan.attributes["llm.token_count.total"]
        == plan.attributes["llm.token_count.completion"] + 10
    )
    assert "You plan work" in plan.attributes["input.value"]
    assert all(s.context.trace_id == root.context.trace_id for s in spans)


async def test_a_confirmed_run_is_two_root_spans(cfg):
    sink, exporter = tracer()
    post = FakeTool("post", risk=RiskClass.EXTERNAL_WRITE)
    agent = agent_for(cfg, [calls(("post", {})), text("posted")], [post], sinks=[sink])
    held = await agent.orchestrator.run("post it")
    roots = [s for s in spans_of(exporter) if s.name == "agent.run"]
    assert roots[0].attributes["bijou.status"] == "awaiting_confirmation"
    await agent.orchestrator.confirm(held.session_id, held.confirmation.token, True)
    confirm = next(s for s in spans_of(exporter) if s.name == "agent.confirm")
    assert confirm.attributes["bijou.status"] == "answered"


async def test_a_pick_and_an_equipped_skill_run_carry_their_timings(cfg):
    sink, exporter = tracer()
    events = [
        TraceEvent(kind=TraceKind.RUN_STARTED, session_id="s", data={"request": "find x"}),
        TraceEvent(
            kind=TraceKind.SKILLS_PICKED,
            session_id="s",
            step_id="1",
            data={"skills": ["json_extract"], "reason": "fits", "duration_ms": 420},
        ),
        TraceEvent(
            kind=TraceKind.TOOL_RESULT,
            session_id="s",
            step_id="1",
            data={
                "tool": "run_skill",
                "ok": True,
                "preview": "{}",
                "duration_ms": 2400,
                "skills": ["json_extract"],
                "generate_ms": 900,
            },
        ),
        TraceEvent(
            kind=TraceKind.STEP_DONE, session_id="s", step_id="1", data={"status": "answered"}
        ),
        TraceEvent(kind=TraceKind.RUN_DONE, session_id="s", data={"status": "answered"}),
    ]
    for event in events:
        sink.emit(event)
    step = next(s for s in spans_of(exporter) if s.name == "step")
    tool = next(s for s in spans_of(exporter) if s.name == "tool")
    assert step.attributes["bijou.pick_ms"] == 420
    assert tool.attributes["bijou.equip_ms"] == 1500
    assert tool.attributes["bijou.generate_ms"] == 900


async def test_a_run_that_did_not_answer_marks_its_span_an_error(cfg):
    sink, exporter = tracer()
    sink.emit(TraceEvent(kind=TraceKind.RUN_STARTED, session_id="s", data={"request": "x"}))
    sink.emit(TraceEvent(kind=TraceKind.RUN_DONE, session_id="s", data={"status": "error"}))
    root = next(s for s in spans_of(exporter) if s.name == "agent.run")
    assert root.status.status_code.name == "ERROR"


def test_no_endpoint_means_no_exporter():
    assert open_tracer(Telemetry()) is None
    opened = open_tracer(Telemetry(otlp_endpoint="http://127.0.0.1:9/v1/traces"))
    assert opened is not None
    opened[1]()


def test_spans_name_the_phoenix_project_they_belong_to():
    opened = open_tracer(
        Telemetry(otlp_endpoint="http://127.0.0.1:9/v1/traces", project_name="bijou")
    )
    assert opened is not None
    sink, shutdown = opened
    resource = sink.tracer.resource
    assert resource.attributes["openinference.project.name"] == "bijou"
    assert resource.attributes["service.name"] == "bijou-engine"
    shutdown()


async def test_the_metrics_port_serves_what_the_agent_counted(cfg):
    registry = new_registry()
    await agent_for(cfg, [text("done")], registry=registry).orchestrator.run("find x")
    port, stop = serve_metrics(registry, "127.0.0.1", 0)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            body = response.read().decode()
    finally:
        stop()
    assert 'bijou_agent_runs_total{status="answered"} 1.0' in body
