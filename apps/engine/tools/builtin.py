"""Built-in tools: the time, a public web page, and earlier sessions."""

from __future__ import annotations

from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx

from engine.core.config import Tools
from engine.core.protocols import SessionStore, Tool
from engine.core.types.agent import RequestContext, RiskClass, ToolDefinition, ToolOutput
from engine.core.types.errors import ToolError

_SKIPPED = {"script", "style", "noscript", "svg", "template", "head"}
_BLOCKS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002 - HTMLParser signature
        if tag in _SKIPPED:
            self.skipping += 1
        elif tag in _BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED and self.skipping:
            self.skipping -= 1
        elif tag in _BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skipping:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """The readable text of a page: no scripts or styles, one line per block, no blank runs."""
    parser = _Text()
    parser.feed(html)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


class CurrentTime:
    """Satisfies Tool."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="current_time",
            description="The current date, weekday and time, in UTC and on this machine.",
        )

    async def call(self, ctx: RequestContext, arguments: dict[str, Any]) -> ToolOutput:  # noqa: ARG002 - protocol signature
        utc = datetime.now(UTC)
        local = utc.astimezone()
        return ToolOutput(text=f"UTC {utc:%A %Y-%m-%d %H:%M}; local {local:%A %Y-%m-%d %H:%M %Z}")


class FetchUrl:
    """GET one public page and return its text, a window at a time. Satisfies Tool."""

    def __init__(self, cfg: Tools, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self.client = client or httpx.AsyncClient(
            follow_redirects=True,
            timeout=cfg.fetch_timeout_secs,
            headers={"User-Agent": cfg.user_agent},
        )

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fetch_url",
            description=(
                "Fetch a public web page and return its readable text, "
                f"{self.cfg.fetch_max_chars} characters at a time. start skips that many "
                "characters, to read further into a long page."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "An http or https address."},
                    "start": {"type": "integer", "description": "Characters to skip."},
                },
                "required": ["url"],
            },
            risk=RiskClass.READ_PUBLIC,
        )

    def allowed(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        domains = [d.lower() for d in self.cfg.fetch_allowed_domains]
        return not domains or any(host == d or host.endswith("." + d) for d in domains)

    async def call(self, ctx: RequestContext, arguments: dict[str, Any]) -> ToolOutput:  # noqa: ARG002 - protocol signature
        url = str(arguments.get("url", ""))
        if urlparse(url).scheme not in ("http", "https"):
            raise ToolError("url must start with http:// or https://")
        if not self.allowed(url):
            raise ToolError(f"{urlparse(url).hostname} is not in agent.tools.fetch_allowed_domains")
        start = arguments.get("start", 0)
        start = start if isinstance(start, int) and start > 0 else 0
        try:
            response = await self.client.get(url)
        except httpx.HTTPError as exc:
            raise ToolError(f"could not fetch {url}: {exc}") from exc
        if response.status_code >= 400:
            raise ToolError(f"{url} returned {response.status_code}")
        kind = response.headers.get("content-type", "")
        if "html" in kind:
            text = html_to_text(response.text)
        elif kind.startswith("text/") or "json" in kind or "xml" in kind:
            text = response.text
        else:
            raise ToolError(f"{url} is {kind or 'of unknown type'}, not text")
        window = text[start : start + self.cfg.fetch_max_chars]
        rest = len(text) - start - len(window)
        if rest > 0:
            window += f"\n[{rest} more characters; call again with start={start + len(window)}]"
        return ToolOutput(text=window, data={"url": str(response.url), "chars": len(text)})


class RecallSessions:
    """Search earlier sessions. Satisfies Tool."""

    def __init__(self, sessions: SessionStore, limit: int = 5) -> None:
        self.sessions = sessions
        self.limit = limit

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="recall_sessions",
            description="Search earlier sessions by keywords; returns their requests and answers.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keywords."}},
                "required": ["query"],
            },
        )

    async def call(self, ctx: RequestContext, arguments: dict[str, Any]) -> ToolOutput:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolError("query is empty")
        found = [s for s in self.sessions.search(query, self.limit + 1) if s.id != ctx.session_id]
        if not found:
            return ToolOutput(text=f"no earlier session matches {query!r}")
        lines = [
            f"[{s.id} {s.created_at:%Y-%m-%d}] {s.request}\n  answer: {s.answer}"
            for s in found[: self.limit]
        ]
        return ToolOutput(text="\n".join(lines))


def builtin_tools(cfg: Tools, sessions: SessionStore | None) -> list[Tool]:
    """The built-in tools switched on in agent.tools."""
    tools: list[Tool] = []
    if cfg.current_time:
        tools.append(CurrentTime())
    if cfg.fetch_url:
        tools.append(FetchUrl(cfg))
    if cfg.recall_sessions and sessions is not None:
        tools.append(RecallSessions(sessions))
    return tools
