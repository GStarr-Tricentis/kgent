from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── test helpers ──────────────────────────────────────────────────────────────

def _mcp_tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = f"Tool {name}"
    t.inputSchema = {"type": "object", "properties": {}}
    return t


def _make_session(tools: list, call_results: list | None = None) -> AsyncMock:
    """Build a mock MCP ClientSession.

    call_results: list of strings (success) or Exception instances (raised).
    """
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=MagicMock(tools=tools))
    if call_results is not None:
        side_effects = [
            r if isinstance(r, Exception) else MagicMock(content=r)
            for r in call_results
        ]
        session.call_tool = AsyncMock(side_effect=side_effects)
    return session


def _mcp_patches(session: AsyncMock):
    """Return a pair of patch context managers for stdio_client and ClientSession.

    Using @asynccontextmanager wrappers ensures AsyncExitStack.enter_async_context()
    sees proper async __aenter__/__aexit__ on the type, not just the instance.
    """
    @asynccontextmanager
    async def _fake_stdio(params):
        yield (MagicMock(), MagicMock())

    @asynccontextmanager
    async def _fake_cs(read, write):
        yield session

    return (
        patch("agent_poc.tools.mcp_adapter.stdio_client", _fake_stdio),
        patch("agent_poc.tools.mcp_adapter.ClientSession", _fake_cs),
    )


# ── import guard ──────────────────────────────────────────────────────────────

def test_import_does_not_raise():
    from agent_poc.tools.mcp_adapter import MCPAdapter, MCP_AVAILABLE  # noqa: F401
    assert isinstance(MCP_AVAILABLE, bool)


# ── MCP_AVAILABLE = False ─────────────────────────────────────────────────────

async def test_mcp_unavailable_connect_is_noop():
    import agent_poc.tools.mcp_adapter as mod
    from agent_poc.tools.mcp_adapter import MCPAdapter
    original = mod.MCP_AVAILABLE
    try:
        mod.MCP_AVAILABLE = False
        adapter = MCPAdapter("srv", "cmd", [])
        await adapter.connect()
        assert adapter.list_tools() == []
    finally:
        mod.MCP_AVAILABLE = original


async def test_mcp_unavailable_call_tool_returns_error_string():
    import agent_poc.tools.mcp_adapter as mod
    from agent_poc.tools.mcp_adapter import MCPAdapter
    original = mod.MCP_AVAILABLE
    try:
        mod.MCP_AVAILABLE = False
        adapter = MCPAdapter("srv", "cmd", [])
        result = await adapter.call_tool("any_tool", {})
        assert isinstance(result, str)
        assert "not installed" in result.lower() or "error" in result.lower()
    finally:
        mod.MCP_AVAILABLE = original


# ── connect() ─────────────────────────────────────────────────────────────────

async def test_connect_populates_tools_correctly():
    from agent_poc.tools.mcp_adapter import MCPAdapter
    tools = [_mcp_tool("alpha"), _mcp_tool("beta"), _mcp_tool("gamma")]
    session = _make_session(tools)
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        await adapter.connect()
    assert len(adapter.list_tools()) == 3
    assert {t.name for t in adapter.list_tools()} == {"alpha", "beta", "gamma"}


async def test_connect_sets_tool_source_to_mcp():
    from agent_poc.agent.types import ToolSource
    from agent_poc.tools.mcp_adapter import MCPAdapter
    session = _make_session([_mcp_tool("t1")])
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        await adapter.connect()
    assert adapter.list_tools()[0].source == ToolSource.MCP


async def test_connect_raises_on_empty_tool_list():
    """RuntimeError when server returns no tools."""
    from agent_poc.tools.mcp_adapter import MCPAdapter
    session = _make_session([])
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        with pytest.raises(RuntimeError, match="no tools"):
            await adapter.connect()


# ── call_tool() ───────────────────────────────────────────────────────────────

async def test_call_tool_returns_expected_string():
    """call_tool() returns str(result.content)."""
    from agent_poc.tools.mcp_adapter import MCPAdapter
    session = _make_session([_mcp_tool("greet")], call_results=["hello world"])
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        await adapter.connect()
        result = await adapter.call_tool("greet", {"name": "Alice"})
    assert result == "hello world"


async def test_call_tool_reconnects_and_retries_on_broken_pipe():
    """BrokenPipeError triggers one reconnect; the retry succeeds."""
    from agent_poc.tools.mcp_adapter import MCPAdapter
    session = _make_session(
        [_mcp_tool("t1")],
        call_results=[BrokenPipeError("pipe broke"), "recovered"],
    )
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        await adapter.connect()
        result = await adapter.call_tool("t1", {})
    assert result == "recovered"
    assert adapter._call_count == 1


async def test_call_tool_does_not_reconnect_on_non_transport_error():
    """ValueError from a tool propagates without triggering a reconnect."""
    from agent_poc.tools.mcp_adapter import MCPAdapter
    session = _make_session([_mcp_tool("t1")])
    session.call_tool = AsyncMock(side_effect=ValueError("bad input"))
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        await adapter.connect()
        with pytest.raises(ValueError, match="bad input"):
            await adapter.call_tool("t1", {})
    # initialize() called exactly once (initial connect); no reconnect fired
    assert session.initialize.call_count == 1


# ── async callable capture ────────────────────────────────────────────────────

async def test_tool_callables_are_bound_to_correct_names():
    """Each registered async callable must invoke call_tool with its own tool name."""
    from agent_poc.tools.mcp_adapter import MCPAdapter
    tools = [_mcp_tool("alpha"), _mcp_tool("beta"), _mcp_tool("gamma")]
    session = _make_session(tools)
    sc, cs = _mcp_patches(session)

    called: list[str] = []

    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        await adapter.connect()

        async def _tracking(name, args):
            called.append(name)
            return "ok"

        adapter.call_tool = _tracking
        for tool in adapter.list_tools():
            await tool.callable({})

    assert sorted(called) == ["alpha", "beta", "gamma"]
    assert len(set(called)) == 3  # no closure variable aliasing


# ── disconnect / context manager ──────────────────────────────────────────────

async def test_disconnect_before_connect_does_not_raise():
    from agent_poc.tools.mcp_adapter import MCPAdapter
    adapter = MCPAdapter("srv", "cmd", [])
    await adapter.disconnect()  # _stack is None — must be a no-op


async def test_context_manager_cleans_up_on_exit():
    from agent_poc.tools.mcp_adapter import MCPAdapter
    session = _make_session([_mcp_tool("t1")])
    sc, cs = _mcp_patches(session)
    with sc, cs:
        async with MCPAdapter("test", "cmd", []) as adapter:
            assert len(adapter.list_tools()) == 1
    assert adapter._session is None
    assert adapter._stack is None
