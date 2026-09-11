"""Composition: the only module that builds concrete implementations and hands them around.

build is pure construction over whatever implementations it is given, which is how tests run a
whole harness over doubles. open_harness builds the real ones, opens every MCP server, and
closes everything on exit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from engine.agent.orchestrator import Orchestrator
from engine.agent.planner import Planner
from engine.agent.policy import RiskPolicy
from engine.agent.selector import SkillSelector
from engine.agent.subagent import Subagent
from engine.agent.toolset import ToolSet
from engine.agent.trace import Collector, Fanout, JsonlTrace
from engine.core.config import AgentConfig, Config
from engine.core.protocols import ChatModel, SessionStore, SkillRuntime, Tool, TraceSink
from engine.model.local_skills import LocalSkillRuntime
from engine.model.openai_compat import OpenAIChat
from engine.model.skill_server import HttpSkillRuntime
from engine.patterns.miner import PatternMiner
from engine.stores.sessions import SqliteSessionStore
from engine.tools.builtin import builtin_tools
from engine.tools.mcp import McpConnection


@dataclass
class Harness:
    """Everything a caller of the harness needs."""

    cfg: AgentConfig
    orchestrator: Orchestrator
    sessions: SessionStore
    runtime: SkillRuntime
    tools: ToolSet
    miner: PatternMiner


def build(
    cfg: AgentConfig,
    *,
    model: ChatModel,
    runtime: SkillRuntime,
    sessions: SessionStore,
    tools: Sequence[Tool],
    sinks: Sequence[TraceSink] = (),
) -> Harness:
    """Wire the loop over the given implementations."""
    collector = Collector()
    every: list[TraceSink] = [collector, *sinks]
    if cfg.trace.enabled:
        every.append(JsonlTrace(cfg.trace.dir))
    trace = Fanout(every)
    toolset = ToolSet(tools).without(cfg.tools.disabled)
    subagent = Subagent(model, toolset, runtime, RiskPolicy(cfg.policy), trace, cfg)
    orchestrator = Orchestrator(
        model=model,
        planner=Planner(model, cfg, trace),
        selector=SkillSelector(model, cfg, trace),
        subagent=subagent,
        runtime=runtime,
        sessions=sessions,
        tools=toolset,
        trace=trace,
        collector=collector,
        cfg=cfg,
    )
    return Harness(
        cfg=cfg,
        orchestrator=orchestrator,
        sessions=sessions,
        runtime=runtime,
        tools=toolset,
        miner=PatternMiner(sessions, cfg.patterns),
    )


def skill_runtime(cfg: Config) -> LocalSkillRuntime | HttpSkillRuntime:
    """The skill bank in this process, or the one at agent.skills.url, by agent.skills.mode."""
    if cfg.agent.skills.mode == "local":
        return LocalSkillRuntime(cfg)
    return HttpSkillRuntime(cfg.agent.skills)


@asynccontextmanager
async def open_harness(cfg: Config) -> AsyncIterator[Harness]:
    """The real agent: the chat model, the skill bank, SQLite, built-in tools and MCP servers."""
    agent = cfg.agent
    async with AsyncExitStack() as stack:
        sessions = SqliteSessionStore(agent.sessions.path)
        stack.callback(sessions.close)
        model = OpenAIChat(agent.llm)
        stack.push_async_callback(model.aclose)
        runtime = skill_runtime(cfg)
        stack.push_async_callback(runtime.aclose)
        tools: list[Tool] = list(builtin_tools(agent.tools, sessions))
        for server in agent.mcp.all_servers():
            connection = McpConnection(server, agent.mcp.max_description_chars)
            tools += await connection.open()
            stack.push_async_callback(connection.close)
        yield build(agent, model=model, runtime=runtime, sessions=sessions, tools=tools)
