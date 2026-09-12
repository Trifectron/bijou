"""engine chat: turns that continue each other, and the JSON-lines protocol the console speaks."""

import io
import json
import sys

from engine.commands.chat import Conversation, LineSink, converse, protocol_write
from engine.core.doubles import (
    FakeSkillRuntime,
    FakeTool,
    MemorySessionStore,
    RoutedModel,
    calls,
    text,
)
from engine.core.types.agent import RiskClass, SkillInfo
from engine.wiring import build


def agent_for(cfg, turns, tools=(), sinks=()):
    worker = iter(turns)
    routes = {
        "You plan work": lambda req: text('{"steps": [{"id": "1", "goal": "g"}]}'),
        "You equip a worker": lambda req: text('{"skills": []}'),
        "You are one worker": lambda req: next(worker),
    }
    return build(
        cfg,
        model=RoutedModel(routes),
        runtime=FakeSkillRuntime([SkillInfo(name="a", description="d", trained=True)]),
        sessions=MemorySessionStore(),
        tools=list(tools),
        sinks=list(sinks),
    )


async def lines(*messages):
    for message in messages:
        yield message if isinstance(message, str) else json.dumps(message)


async def test_each_turn_continues_the_last_until_a_new_conversation(cfg):
    agent = agent_for(cfg, [text("one"), text("two"), text("three")])
    convo = Conversation(agent, "ash")
    first = await convo.ask("hello")
    second = await convo.ask("again")
    assert agent.sessions.get(second.session_id).resumed_from == first.session_id
    convo.new()
    third = await convo.ask("fresh")
    assert agent.sessions.get(third.session_id).resumed_from is None
    assert third.answer == "three"


async def test_the_console_protocol_streams_events_before_each_result(cfg):
    out = []
    post = FakeTool("post", risk=RiskClass.EXTERNAL_WRITE)
    agent = agent_for(cfg, [calls(("post", {})), text("posted")], [post], [LineSink(out.append)])
    ask = {"op": "ask", "text": "post it"}
    await converse(
        Conversation(agent, "local"), lines(ask, {"op": "confirm", "approve": True}), out.append
    )
    kinds = [m["type"] for m in out]
    assert kinds.count("result") == 2
    assert "event" in kinds[: kinds.index("result")]
    held = out[kinds.index("result")]["result"]
    assert held["status"] == "awaiting_confirmation" and "events" not in held
    assert out[-1]["result"]["answer"] == "posted"
    assert json.loads(json.dumps(out)) == out


async def test_a_bad_line_gets_an_error_and_the_conversation_goes_on(cfg):
    out = []
    convo = Conversation(agent_for(cfg, [text("fine")]), "local")
    bad = ["not json", "[1, 2]", {"op": "dance"}, {"op": "ask"}, {"op": "confirm"}]
    await converse(convo, lines(*bad, {"op": "ask", "text": "hi"}), out.append)
    assert [m["type"] for m in out] == ["error"] * 5 + ["result"]
    assert "expected an object" in out[1]["message"]
    assert "unknown op 'dance'" in out[2]["message"]
    assert "ask needs text" in out[3]["message"]
    assert "nothing is waiting" in out[4]["message"]
    assert out[-1]["result"]["answer"] == "fine"


async def test_stats_answers_with_what_the_agent_has_counted(cfg):
    out = []
    convo = Conversation(agent_for(cfg, [text("done")]), "local")
    await converse(convo, lines({"op": "ask", "text": "hi"}, {"op": "stats"}), out.append)
    counted = out[-1]
    assert counted["type"] == "stats"
    assert counted["stats"]["bijou_agent_runs_total{status=answered}"] == 1
    assert counted["stats"]["bijou_agent_run_seconds_count"] == 1


def test_only_protocol_lines_reach_stdout(monkeypatch):
    out, printed = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", printed)
    write = protocol_write()
    sys.stdout.write("chatter from anything that writes to stdout\n")
    write({"type": "ready", "metrics": None})
    assert sys.stdout is printed
    assert out.getvalue() == '{"type": "ready", "metrics": null}\n'
    assert "chatter" in printed.getvalue()
