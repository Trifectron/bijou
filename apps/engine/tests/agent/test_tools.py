"""Built-in tools, the tool set, the policy, and MCP servers through the real SDK in-process."""

import httpx
import pytest

from engine.agent.policy import RiskPolicy, payload_hash
from engine.agent.toolset import ToolSet
from engine.core.config import McpServer, Policy, Tools
from engine.core.doubles import FakeTool, MemorySessionStore
from engine.core.types.agent import ProposedAction, RiskClass
from engine.core.types.errors import ConfigError, ToolError
from engine.tools.builtin import CurrentTime, FetchUrl, RecallSessions, builtin_tools, html_to_text
from engine.tools.mcp import McpConnection, compact, risk_for, tool_name

PAGE = """<html><head><title>t</title><style>x{}</style></head>
<body><h1>Jobs</h1><script>var a=1;</script>
<p>Applied   Scientist</p><li>Seattle</li></body></html>"""


def test_html_becomes_readable_text():
    assert html_to_text(PAGE) == "Jobs\nApplied Scientist\nSeattle"


def fetcher(handler, **cfg):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return FetchUrl(Tools(**cfg), client)


async def test_fetch_url_reads_a_page_a_window_at_a_time(ctx):
    tool = fetcher(lambda r: httpx.Response(200, html=PAGE), fetch_max_chars=5)
    first = await tool.call(ctx, {"url": "https://jobs.example/x"})
    assert first.text.startswith("Jobs\n")
    assert "call again with start=5" in first.text
    rest = await tool.call(ctx, {"url": "https://jobs.example/x", "start": 5})
    assert rest.text.startswith("Appli")


async def test_fetch_url_refuses_bad_schemes_other_domains_and_binary(ctx):
    tool = fetcher(
        lambda r: httpx.Response(200, content=b"\x00", headers={"content-type": "image/png"}),
        fetch_allowed_domains=["example.com"],
    )
    with pytest.raises(ToolError, match="http"):
        await tool.call(ctx, {"url": "file:///etc/passwd"})
    with pytest.raises(ToolError, match="fetch_allowed_domains"):
        await tool.call(ctx, {"url": "https://evil.test/"})
    with pytest.raises(ToolError, match="not text"):
        await tool.call(ctx, {"url": "https://jobs.example.com/logo"})


async def test_fetch_url_reports_http_errors(ctx):
    tool = fetcher(lambda r: httpx.Response(404))
    with pytest.raises(ToolError, match="404"):
        await tool.call(ctx, {"url": "https://x.test/"})


async def test_recall_finds_other_sessions_only(ctx):
    from datetime import UTC, datetime

    from engine.core.types.agent import RunStatus, SessionRecord

    def record(session_id, request):
        at = datetime(2026, 9, 1, tzinfo=UTC)
        return SessionRecord(
            id=session_id, created_at=at, updated_at=at, request=request, status=RunStatus.ANSWERED
        )

    store = MemorySessionStore()
    store.save(record("s1", "amazon jobs"))
    store.save(record(ctx.session_id, "amazon jobs again"))
    found = await RecallSessions(store).call(ctx, {"query": "amazon"})
    assert "[s1" in found.text and ctx.session_id not in found.text


async def test_current_time_names_the_weekday(ctx):
    out = await CurrentTime().call(ctx, {})
    assert out.text.startswith("UTC ") and "day" in out.text


def test_builtin_tools_follow_the_switches():
    names = [t.definition.name for t in builtin_tools(Tools(fetch_url=False), MemorySessionStore())]
    assert names == ["current_time", "recall_sessions"]
    assert [t.definition.name for t in builtin_tools(Tools(), None)] == [
        "current_time",
        "fetch_url",
    ]


def test_tool_names_are_unique():
    with pytest.raises(ConfigError, match="two tools"):
        ToolSet([FakeTool("a"), FakeTool("a")])
    assert ToolSet([FakeTool("a"), FakeTool("b")]).without(["a"]).names() == ["b"]


def test_the_policy_by_risk_class():
    policy = RiskPolicy(Policy())

    def decide(risk):
        return policy.authorize(ProposedAction(tool="t", arguments={"a": 1}, risk=risk)).kind

    assert decide(RiskClass.READ_PUBLIC) == "allow"
    assert decide(RiskClass.PREPARE_WRITE) == "allow"
    assert decide(RiskClass.READ_AUTHENTICATED) == "deny"
    assert decide(RiskClass.EXTERNAL_WRITE) == "confirm"
    assert decide(RiskClass.DESTRUCTIVE) == "confirm"
    assert decide(RiskClass.FORBIDDEN) == "deny"
    stricter = RiskPolicy(Policy(confirm_from=RiskClass.PREPARE_WRITE))
    held = stricter.authorize(ProposedAction(tool="t", arguments={}, risk=RiskClass.PREPARE_WRITE))
    assert held.kind == "confirm"


def test_payload_hashes_ignore_key_order():
    assert payload_hash({"a": 1, "b": 2}) == payload_hash({"b": 2, "a": 1})
    assert payload_hash({"a": 1}) != payload_hash({"a": 2})


@pytest.mark.parametrize(
    ("name", "risk"),
    [
        ("browser_navigate", RiskClass.READ_PUBLIC),
        ("browser_snapshot", RiskClass.READ_PUBLIC),
        ("browser_type", RiskClass.PREPARE_WRITE),
        ("browser_click", RiskClass.EXTERNAL_WRITE),
        ("browser_evaluate", RiskClass.EXTERNAL_WRITE),
        ("something_unknown", RiskClass.EXTERNAL_WRITE),
    ],
)
def test_mcp_risk_by_tool_name(name, risk):
    assert risk_for(name) is risk


def test_mcp_names_and_schemas():
    assert tool_name("browser", "navigate.to") == "browser_navigate_to"
    schema = {"type": "object", "properties": {"url": {}, "timeout": {}}, "required": ["url"]}
    assert compact(schema, True)["properties"] == {"url": {}}
    assert compact(schema, False) == schema


def test_an_mcp_server_needs_one_transport():
    with pytest.raises(ConfigError, match="exactly one"):
        McpServer(name="x")
    with pytest.raises(ConfigError, match="exactly one"):
        McpServer(name="x", url="http://a", command="b")


async def test_an_mcp_server_becomes_tools_through_the_sdk(ctx):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("pages")

    @server.tool()
    def read_page(url: str, verbose: bool = False) -> str:
        """Read a page."""
        return f"contents of {url}"

    @server.tool()
    def fail_loudly() -> str:
        """Always fails."""
        raise ValueError("nope")

    connection = McpConnection(McpServer(name="pages", command="unused"), target=server)
    tools = {t.definition.name: t for t in await connection.open()}
    try:
        read = tools["pages_read_page"]
        assert read.definition.risk is RiskClass.READ_PUBLIC
        assert read.definition.source == "mcp:pages"
        assert set(read.definition.parameters["properties"]) == {"url"}
        out = await read.call(ctx, {"url": "https://x.test"})
        assert out.text == "contents of https://x.test"
        assert tools["pages_fail_loudly"].definition.risk is RiskClass.EXTERNAL_WRITE
        with pytest.raises(ToolError):
            await tools["pages_fail_loudly"].call(ctx, {})
    finally:
        await connection.close()


async def test_an_mcp_allowlist_limits_the_tools(ctx):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("s")

    @server.tool()
    def one() -> str:
        """One."""
        return "1"

    @server.tool()
    def two() -> str:
        """Two."""
        return "2"

    connection = McpConnection(McpServer(name="s", command="x", tools=["two"]), target=server)
    try:
        assert [t.definition.name for t in await connection.open()] == ["s_two"]
    finally:
        await connection.close()


async def test_an_unreachable_mcp_server_names_itself():
    connection = McpConnection(
        McpServer(name="browser", url="http://127.0.0.1:9/mcp", timeout_secs=2)
    )
    with pytest.raises(ConfigError, match="browser"):
        await connection.open()
