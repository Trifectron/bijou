"""The HTTP surface over a harness of doubles, and the configuration's validation."""

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from engine.core.config import AgentConfig, Config
from engine.core.doubles import (
    FakeSkillRuntime,
    FakeTool,
    MemorySessionStore,
    RoutedModel,
    calls,
    text,
)
from engine.core.types.agent import RiskClass, SkillInfo
from engine.core.types.errors import ConfigError
from engine.routes.agent import create_app
from engine.wiring import build


def client_for(cfg, routes, tools=()):
    @asynccontextmanager
    async def factory(c):
        yield build(
            c,
            model=RoutedModel(routes),
            runtime=FakeSkillRuntime([SkillInfo(name="a", description="d", trained=True)]),
            sessions=MemorySessionStore(),
            tools=list(tools),
        )

    return TestClient(create_app(cfg, factory))


ROUTES = {
    "You plan work": lambda req: text('{"steps": [{"id": "1", "goal": "g"}]}'),
    "You equip a worker": lambda req: text('{"skills": []}'),
    "You are one worker": lambda req: text("hello"),
}


def test_run_then_find_the_session(cfg):
    with client_for(cfg, ROUTES) as client:
        assert client.get("/health").json()["trained_skills"] == 1
        result = client.post("/run", json={"request": "say hello"}).json()
        assert result["status"] == "answered" and result["answer"] == "hello"
        assert result["events"]
        listed = client.get("/sessions").json()
        assert listed[0]["id"] == result["session_id"]
        assert (
            client.get("/sessions", params={"q": "hello"}).json()[0]["id"] == result["session_id"]
        )
        assert client.get(f"/sessions/{result['session_id']}").json()["request"] == "say hello"
        assert client.get("/sessions/nope").status_code == 404
        assert client.get("/skills").json()[0]["name"] == "a"
        assert client.get("/patterns").json() == []


def test_confirmation_over_http(cfg):
    post = FakeTool("post", risk=RiskClass.EXTERNAL_WRITE)
    turns = iter([calls(("post", {})), text("done")])
    routes = ROUTES | {"You are one worker": lambda req: next(turns)}
    with client_for(cfg, routes, [post]) as client:
        held = client.post("/run", json={"request": "post"}).json()
        assert held["status"] == "awaiting_confirmation"
        body = {"session_id": held["session_id"], "token": "wrong"}
        assert client.post("/confirm", json=body).status_code == 409
        body["token"] = held["confirmation"]["token"]
        assert client.post("/confirm", json=body).json()["answer"] == "done"


def test_bad_bodies_and_unknown_resumes(cfg):
    with client_for(cfg, ROUTES) as client:
        assert client.post("/run", json={"request": ""}).status_code == 422
        assert client.post("/run", json={"request": "x", "resume_from": "nope"}).status_code == 404


def test_defaults_load_and_only_harness_keys_are_read(tmp_path, monkeypatch):
    path = tmp_path / "bijou.toml"
    path.write_text("[train]\nlr = 1\n[agent.loop]\nmax_turns = 3\n[agent.llm]\nmodel = 'm'\n")
    monkeypatch.setenv("BIJOU_CONFIG_FILE", str(path))
    monkeypatch.setenv("BIJOU_AGENT__LLM__API_KEY", "secret")
    cfg = Config().agent
    assert cfg.loop.max_turns == 3 and cfg.llm.model == "m"
    assert cfg.llm.api_key.get_secret_value() == "secret"
    assert "secret" not in cfg.model_dump_json()


def test_the_committed_file_loads():
    assert Config().agent.http.port > 0


@pytest.mark.parametrize(
    ("table", "values", "message"),
    [
        ("loop", {"max_turns": 0}, "max_turns"),
        ("planning", {"concurrency": 0}, "concurrency"),
        ("patterns", {"similarity": 0}, "similarity"),
        ("policy", {"confirm_from": "read_public"}, "every tool"),
    ],
)
def test_contradictions_are_rejected_at_load(table, values, message):
    with pytest.raises(ConfigError, match=message):
        AgentConfig(**{table: values})


def test_two_mcp_servers_cannot_share_a_name():
    with pytest.raises(ConfigError, match="two servers named browser"):
        AgentConfig(
            mcp={
                "playwright_url": "http://localhost:8931/mcp",
                "servers": [{"name": "browser", "url": "http://x"}],
            }
        )
