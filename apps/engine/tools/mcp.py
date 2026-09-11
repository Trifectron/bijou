"""MCP servers as tools, through the official SDK's Client.

One connection per configured server, over Streamable HTTP or stdio. Every tool it lists becomes
a Tool named server_tool, with a risk class derived from the tool's name unless the server's
config sets one. The Playwright MCP server is the browser; a computer-use server is configured
the same way.
"""

from __future__ import annotations

import re
from contextlib import AsyncExitStack
from typing import Any

from mcp import Client, StdioServerParameters

from engine.core.config import McpServer
from engine.core.types.agent import RequestContext, RiskClass, ToolDefinition, ToolOutput
from engine.core.types.errors import ConfigError, ToolError

# Look at a page or a screen without changing it.
READS = (
    "navigate",
    "snapshot",
    "screenshot",
    "find",
    "tabs",
    "wait_for",
    "console",
    "network_requests",
    "resize",
    "install",
    "list",
    "get",
    "read",
    "search",
)
# Change what a page holds without committing it. Reversible by navigating away.
DRAFTS = ("type", "fill_form", "select_option", "hover", "drag")
# Can commit a page, run code in it, or act on the machine.
COMMITS = ("submit", "click", "press_key", "evaluate", "file_upload", "handle_dialog", "run")


def risk_for(name: str) -> RiskClass:
    """Risk by name. Unrecognised tools are treated as consequential until classified."""
    lowered = name.lower()
    if any(k in lowered for k in COMMITS):
        return RiskClass.EXTERNAL_WRITE
    if any(k in lowered for k in DRAFTS):
        return RiskClass.PREPARE_WRITE
    if any(k in lowered for k in READS):
        return RiskClass.READ_PUBLIC
    return RiskClass.EXTERNAL_WRITE


def tool_name(server: str, remote: str) -> str:
    """The name the model sees: server_tool, in the characters tool names allow."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}_{remote}")[:64]


def compact(schema: dict[str, Any], required_only: bool) -> dict[str, Any]:
    """The input schema, with optional properties dropped when required_only is set."""
    if not required_only:
        return schema
    required = set(schema.get("required") or [])
    properties = schema.get("properties") or {}
    return {**schema, "properties": {k: v for k, v in properties.items() if k in required}}


def text_of(content: list[Any]) -> str:
    """The text blocks of a tool result, with a marker for anything that is not text."""
    parts = []
    for block in content:
        kind = getattr(block, "type", "")
        parts.append(block.text if kind == "text" else f"[{kind or 'unknown'} content]")
    return "\n".join(parts)


class McpTool:
    """One remote tool. Satisfies Tool."""

    def __init__(self, connection: McpConnection, remote: str, definition: ToolDefinition) -> None:
        self.connection = connection
        self.remote = remote
        self._definition = definition

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def call(self, ctx: RequestContext, arguments: dict[str, Any]) -> ToolOutput:
        timeout = max(min(self.connection.server.timeout_secs, ctx.remaining()), 0.001)
        return ToolOutput(text=await self.connection.call(self.remote, arguments, timeout))


class McpConnection:
    """One server's session, held open for the life of the harness."""

    def __init__(
        self, server: McpServer, max_description_chars: int = 200, target: Any = None
    ) -> None:
        self.server = server
        self.max_description_chars = max_description_chars
        self.target = target
        self._stack = AsyncExitStack()
        self._client: Client | None = None

    def _target(self) -> Any:
        if self.target is not None:
            return self.target
        if self.server.url:
            return self.server.url
        return StdioServerParameters(command=self.server.command, args=self.server.args)

    async def open(self) -> list[McpTool]:
        """Connect, list the server's tools, and wrap the allowed ones."""
        where = self.server.url or self.server.command
        try:
            client = Client(self._target(), read_timeout_seconds=self.server.timeout_secs)
            self._client = await self._stack.enter_async_context(client)
            listed = []
            cursor: str | None = None
            while True:
                page = await self._client.list_tools(cursor=cursor)
                listed += page.tools
                cursor = getattr(page, "next_cursor", None)
                if not cursor:
                    break
        except Exception as exc:  # any failure of the remote server
            await self._stack.aclose()
            raise ConfigError(
                f"MCP server {self.server.name} at {where} could not be opened: {exc}; "
                "start it, or remove it from agent.mcp"
            ) from exc
        allowed = set(self.server.tools)
        tools = []
        for remote in listed:
            if allowed and remote.name not in allowed:
                continue
            description = (remote.description or remote.name)[: self.max_description_chars]
            definition = ToolDefinition(
                name=tool_name(self.server.name, remote.name),
                description=description,
                parameters=compact(dict(remote.input_schema), self.server.required_props_only),
                risk=self.server.risk or risk_for(remote.name),
                sequential=True,
                source=f"mcp:{self.server.name}",
            )
            tools.append(McpTool(self, remote.name, definition))
        return tools

    async def call(self, remote: str, arguments: dict[str, Any], timeout: float) -> str:
        if self._client is None:
            raise ToolError(f"MCP server {self.server.name} is not open")
        try:
            result = await self._client.call_tool(remote, arguments, read_timeout_seconds=timeout)
        except Exception as exc:  # any failure of the remote server
            raise ToolError(f"{self.server.name} {remote} failed: {exc}") from exc
        text = text_of(list(result.content))
        if result.is_error:
            raise ToolError(text or f"{self.server.name} {remote} reported an error")
        return text

    async def close(self) -> None:
        await self._stack.aclose()
        self._client = None
