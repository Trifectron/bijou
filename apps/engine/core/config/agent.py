"""The agent's settings: the [agent.*] tables of bijou.toml.

Secrets and per-machine URLs live in .env as BIJOU_AGENT__<TABLE>__<KEY>. A combination the agent
cannot serve is rejected at load rather than clamped at use.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from engine.core.types.agent import RiskClass
from engine.core.types.errors import ConfigError

_SERVER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class _Table(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Llm(_Table):
    """The chat model: an OpenAI-compatible endpoint, llama-server by default."""

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: SecretStr = SecretStr("")
    model: str = "Qwen/Qwen3-4B-GGUF:Q4_K_M"
    thinking: bool = False
    max_tokens: int = 1024
    temperature: float = 0.2
    timeout_secs: float = 120.0


class SkillServer(_Table):
    """Where the skill bank runs: in this process, or at url, served by engine serve-skills."""

    mode: Literal["local", "http"] = "local"
    url: str = "http://127.0.0.1:8100"
    timeout_secs: float = 300.0
    gen_length: int | None = None
    steps: int | None = None


class Loop(_Table):
    """One subagent's tool loop, and the run around it."""

    max_turns: int = 8
    max_model_retries: int = 2
    retry_base_ms: int = 250
    retry_cap_ms: int = 8000
    run_timeout_secs: float = 600.0
    tool_timeout_secs: float = 60.0
    max_tool_output_chars: int = 6000


class Planning(_Table):
    """How a request is split into steps and what each step may equip."""

    max_steps: int = 4
    concurrency: int = 2
    max_skills: int = 2
    allow_schedules: bool = True


class Policy(_Table):
    """What may run. Classes at or above confirm_from wait for the user."""

    confirm_from: RiskClass = RiskClass.EXTERNAL_WRITE
    allow_authenticated_reads: bool = False
    confirmation_ttl_secs: int = 900


class Tools(_Table):
    """Built-in tools. disabled removes any tool by name, MCP ones included."""

    disabled: list[str] = Field(default_factory=list)
    current_time: bool = True
    recall_sessions: bool = True
    fetch_url: bool = True
    fetch_max_chars: int = 8000
    fetch_timeout_secs: float = 20.0
    # Empty allows every domain.
    fetch_allowed_domains: list[str] = Field(default_factory=list)
    user_agent: str = "bijou-engine/0.1"


class McpServer(_Table):
    """One MCP server, over Streamable HTTP (url) or stdio (command and args)."""

    name: str
    url: str = ""
    command: str = ""
    args: list[str] = Field(default_factory=list)
    # Empty exposes every tool the server lists.
    tools: list[str] = Field(default_factory=list)
    timeout_secs: float = 60.0
    required_props_only: bool = True
    # Overrides the risk derived from each tool's name, for a server trusted as a whole.
    risk: RiskClass | None = None

    @model_validator(mode="after")
    def _check(self) -> McpServer:
        if not _SERVER_NAME.match(self.name):
            raise ConfigError(f"mcp server name {self.name!r} must be lower snake case")
        if bool(self.url) == bool(self.command):
            raise ConfigError(f"mcp server {self.name} needs exactly one of url and command")
        return self


class Mcp(_Table):
    """MCP servers. playwright_url is per-machine and lives in .env; set, it adds browser."""

    servers: list[McpServer] = Field(default_factory=list)
    playwright_url: str = ""
    max_description_chars: int = 200

    def all_servers(self) -> list[McpServer]:
        """The configured servers, plus the Playwright browser when its URL is set."""
        servers = list(self.servers)
        if self.playwright_url:
            servers.append(McpServer(name="browser", url=self.playwright_url))
        return servers


class Sessions(_Table):
    """The session index."""

    path: Path = Path(".bijou/sessions.db")
    resume_chars: int = 2000
    list_limit: int = 20


class Patterns(_Table):
    """The miner that proposes skills from recurring work no skill covered."""

    min_occurrences: int = 3
    similarity: float = 0.5
    window: int = 500
    proposals_dir: Path = Path("data/proposals")
    max_examples: int = 20


class Trace(_Table):
    """One JSONL file per session."""

    enabled: bool = True
    dir: Path = Path(".bijou/traces")


class Http(_Table):
    """The agent's HTTP surface, engine serve."""

    host: str = "127.0.0.1"
    port: int = 8200


class Prompt(_Table):
    """The wording of every prompt the agent writes. Changing it changes behaviour."""

    planner: str = (
        "You plan work for an agent. Split the user's request into the fewest steps that each "
        "one worker can finish on its own, at most {max_steps}. A request one worker can handle "
        "is one step. Give each step an id and a goal written so it makes sense without the "
        "others; depends_on lists the ids of earlier steps whose results it needs.\n\n"
        "Workers have these tools: {tools}\n"
        "Workers can equip these skills: {skills}"
    )
    selector: str = (
        "You equip a worker for one step. Skills are small specialised models; equip one only "
        "when the step's work is what the skill describes, at most {max_skills}, and none when "
        "no skill fits. {schedule_hint}\n\nSkills:\n{skills}"
    )
    schedule_hint: str = (
        "When two skills are needed, a schedule may give one the early part of generation, "
        "where structure is settled, and the other the late part, where details are; phases "
        "are fractions in [0, 1) and should split at 0.5."
    )
    subagent: str = (
        "You are one worker in a larger task. Finish your step using the tools you have, then "
        "reply with the result and nothing else. Call a tool only when it gets you something "
        "you do not already have. If a tool is denied or fails, work with what you have. "
        "Never invent facts a tool did not give you."
    )
    equipped: str = (
        "You have these skills equipped. run_skill sends text to a model specialised in them "
        "and returns its output:\n{skills}"
    )
    synthesis: str = (
        "Several workers each finished one step of the user's request. Write the answer to the "
        "request from their results. Say plainly what failed or is missing. Do not add facts "
        "the results do not contain."
    )


class AgentConfig(_Table):
    """Everything under [agent]."""

    llm: Llm = Field(default_factory=Llm)
    skills: SkillServer = Field(default_factory=SkillServer)
    loop: Loop = Field(default_factory=Loop)
    planning: Planning = Field(default_factory=Planning)
    policy: Policy = Field(default_factory=Policy)
    tools: Tools = Field(default_factory=Tools)
    mcp: Mcp = Field(default_factory=Mcp)
    sessions: Sessions = Field(default_factory=Sessions)
    patterns: Patterns = Field(default_factory=Patterns)
    trace: Trace = Field(default_factory=Trace)
    http: Http = Field(default_factory=Http)
    prompt: Prompt = Field(default_factory=Prompt)

    @model_validator(mode="after")
    def validate_combinations(self) -> AgentConfig:
        positive = {
            "loop.max_turns": self.loop.max_turns,
            "loop.run_timeout_secs": self.loop.run_timeout_secs,
            "loop.tool_timeout_secs": self.loop.tool_timeout_secs,
            "loop.max_tool_output_chars": self.loop.max_tool_output_chars,
            "planning.max_steps": self.planning.max_steps,
            "planning.concurrency": self.planning.concurrency,
            "policy.confirmation_ttl_secs": self.policy.confirmation_ttl_secs,
            "patterns.min_occurrences": self.patterns.min_occurrences,
        }
        for key, value in positive.items():
            if value <= 0:
                raise ConfigError(f"agent.{key} must be positive")
        if self.planning.max_skills < 0:
            raise ConfigError("agent.planning.max_skills cannot be negative")
        if self.loop.max_model_retries < 0:
            raise ConfigError("agent.loop.max_model_retries cannot be negative")
        if not 0 < self.patterns.similarity <= 1:
            raise ConfigError("agent.patterns.similarity must lie in (0, 1]")
        if self.policy.confirm_from is RiskClass.READ_PUBLIC:
            raise ConfigError("agent.policy.confirm_from read_public would hold every tool")
        names = [s.name for s in self.mcp.all_servers()]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise ConfigError(f"agent.mcp has two servers named {', '.join(duplicated)}")
        return self
