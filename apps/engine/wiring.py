"""Composition: the only module that builds concrete implementations and hands them around.

build is pure construction over whatever implementations it is given, which is how tests run a
whole agent over doubles. open_agent builds the real ones, opens every MCP server and the span
exporter, and closes everything on exit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from prometheus_client import CollectorRegistry

from engine.agent.orchestrator import Orchestrator
from engine.agent.planner import Planner
from engine.agent.policy import RiskPolicy
from engine.agent.selector import SkillSelector
from engine.agent.subagent import Subagent
from engine.agent.toolset import ToolSet
from engine.agent.trace import Collector, Fanout, JsonlTrace
from engine.clients.chat import OpenAIChat
from engine.clients.local_bank import LocalBank
from engine.core.config import AgentConfig, Config
from engine.core.protocols import ChatModel, SessionStore, SkillRuntime, Tool, TraceSink
from engine.memory.patterns import PatternMiner
from engine.memory.sessions import SqliteSessionStore
from engine.telemetry.metrics import AgentMetrics, BankMetrics, MetricsSink, new_registry
from engine.telemetry.otel import open_tracer
from engine.tools.builtin import builtin_tools
from engine.tools.mcp import McpConnection


@dataclass
class Agent:
    """Everything a caller of the agent needs."""

    cfg: AgentConfig
    orchestrator: Orchestrator
    sessions: SessionStore
    runtime: SkillRuntime
    tools: ToolSet
    miner: PatternMiner
    registry: CollectorRegistry


def build(
    cfg: AgentConfig,
    *,
    model: ChatModel,
    runtime: SkillRuntime,
    sessions: SessionStore,
    tools: Sequence[Tool],
    sinks: Sequence[TraceSink] = (),
    registry: CollectorRegistry | None = None,
) -> Agent:
    """Wire the loop over the given implementations."""
    registry = registry or new_registry()
    collector = Collector()
    every: list[TraceSink] = [collector, MetricsSink(AgentMetrics(registry)), *sinks]
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
    return Agent(
        cfg=cfg,
        orchestrator=orchestrator,
        sessions=sessions,
        runtime=runtime,
        tools=toolset,
        miner=PatternMiner(sessions, cfg.patterns),
        registry=registry,
    )


@asynccontextmanager
async def open_agent(cfg: Config, sinks: Sequence[TraceSink] = ()) -> AsyncIterator[Agent]:
    """The real agent: the chat model, the skill bank in this process, SQLite, tools, MCP servers,
    telemetry. sinks see every trace event too, which is how engine chat streams a run."""
    agent_cfg = cfg.agent
    registry = new_registry()
    async with AsyncExitStack() as stack:
        every: list[TraceSink] = list(sinks)
        tracer = open_tracer(cfg.telemetry)
        if tracer is not None:
            sink, shutdown = tracer
            every.append(sink)
            stack.callback(shutdown)
        sessions = SqliteSessionStore(agent_cfg.sessions.path)
        stack.callback(sessions.close)
        model = OpenAIChat(agent_cfg.llm)
        stack.push_async_callback(model.aclose)
        runtime = LocalBank(cfg, metrics=BankMetrics(registry))
        stack.push_async_callback(runtime.aclose)
        tools: list[Tool] = list(builtin_tools(agent_cfg.tools, sessions))
        for server in agent_cfg.mcp.servers:
            connection = McpConnection(server, agent_cfg.mcp.max_description_chars)
            tools += await connection.open()
            stack.push_async_callback(connection.close)
        yield build(
            agent_cfg,
            model=model,
            runtime=runtime,
            sessions=sessions,
            tools=tools,
            sinks=every,
            registry=registry,
        )
