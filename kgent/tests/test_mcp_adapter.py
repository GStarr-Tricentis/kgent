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
        patch("kgent.tools.mcp_adapter.stdio_client", _fake_stdio),
        patch("kgent.tools.mcp_adapter.ClientSession", _fake_cs),
    )


# ── import guard ──────────────────────────────────────────────────────────────

def test_import_does_not_raise():
    from kgent.tools.mcp_adapter import MCPAdapter, MCP_AVAILABLE  # noqa: F401
    assert isinstance(MCP_AVAILABLE, bool)


# ── MCP_AVAILABLE = False ─────────────────────────────────────────────────────

async def test_mcp_unavailable_connect_is_noop():
    import kgent.tools.mcp_adapter as mod
    from kgent.tools.mcp_adapter import MCPAdapter
    original = mod.MCP_AVAILABLE
    try:
        mod.MCP_AVAILABLE = False
        adapter = MCPAdapter("srv", "cmd", [])
        await adapter.connect()
        assert adapter.list_tools() == []
    finally:
        mod.MCP_AVAILABLE = original


async def test_mcp_unavailable_call_tool_returns_error_string():
    import kgent.tools.mcp_adapter as mod
    from kgent.tools.mcp_adapter import MCPAdapter
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
    from kgent.tools.mcp_adapter import MCPAdapter
    tools = [_mcp_tool("alpha"), _mcp_tool("beta"), _mcp_tool("gamma")]
    session = _make_session(tools)
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        await adapter.connect()
    assert len(adapter.list_tools()) == 3
    assert {t.name for t in adapter.list_tools()} == {"alpha", "beta", "gamma"}


async def test_connect_sets_tool_source_to_mcp():
    from kgent.agent.types import ToolSource
    from kgent.tools.mcp_adapter import MCPAdapter
    session = _make_session([_mcp_tool("t1")])
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        await adapter.connect()
    assert adapter.list_tools()[0].source == ToolSource.MCP


async def test_connect_raises_on_empty_tool_list():
    """RuntimeError when server returns no tools."""
    from kgent.tools.mcp_adapter import MCPAdapter
    session = _make_session([])
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        with pytest.raises(RuntimeError, match="no tools"):
            await adapter.connect()


# ── call_tool() ───────────────────────────────────────────────────────────────

async def test_call_tool_returns_expected_string():
    """call_tool() returns str(result.content)."""
    from kgent.tools.mcp_adapter import MCPAdapter
    session = _make_session([_mcp_tool("greet")], call_results=["hello world"])
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        await adapter.connect()
        result = await adapter.call_tool("greet", {"name": "Alice"})
    assert result == "hello world"


async def test_call_tool_raises_on_broken_pipe():
    """BrokenPipeError propagates out of call_tool(); _mcp_loop owns reconnection."""
    from kgent.tools.mcp_adapter import MCPAdapter
    session = _make_session(
        [_mcp_tool("t1")],
        call_results=[BrokenPipeError("pipe broke")],
    )
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [])
        await adapter.connect()
        with pytest.raises(BrokenPipeError, match="pipe broke"):
            await adapter.call_tool("t1", {})
    assert adapter._call_count == 0


async def test_call_tool_does_not_reconnect_on_non_transport_error():
    """ValueError from a tool propagates without triggering a reconnect."""
    from kgent.tools.mcp_adapter import MCPAdapter
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
    from kgent.tools.mcp_adapter import MCPAdapter
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
    from kgent.tools.mcp_adapter import MCPAdapter
    adapter = MCPAdapter("srv", "cmd", [])
    await adapter.disconnect()  # _tg is None — must be a no-op


async def test_context_manager_cleans_up_on_exit():
    from kgent.tools.mcp_adapter import MCPAdapter
    session = _make_session([_mcp_tool("t1")])
    sc, cs = _mcp_patches(session)
    with sc, cs:
        async with MCPAdapter("test", "cmd", []) as adapter:
            assert len(adapter.list_tools()) == 1
    assert adapter._session is None
    assert adapter._tg is None


# ── Fix 2: max_calls wiring ───────────────────────────────────────────────────

async def test_mcp_adapter_accepts_max_calls_constructor_param():
    """MCPAdapter.__init__ must accept a max_calls kwarg and store it as _max_calls."""
    from kgent.tools.mcp_adapter import MCPAdapter
    adapter = MCPAdapter("srv", "cmd", [], max_calls=50)
    assert adapter._max_calls == 50


async def test_proactive_reconnect_fires_at_max_calls():
    """When _call_count reaches _max_calls, reconnect fires before the next call."""
    from kgent.tools.mcp_adapter import MCPAdapter
    session = _make_session([_mcp_tool("t1")], call_results=["a", "b"])
    sc, cs = _mcp_patches(session)
    with sc, cs:
        adapter = MCPAdapter("test", "cmd", [], max_calls=1)
        await adapter.connect()
        r1 = await adapter.call_tool("t1", {})
        # _call_count is now 1 == max_calls; next call must trigger proactive reconnect
        r2 = await adapter.call_tool("t1", {})

    assert r1 == "a"
    assert r2 == "b"
    # initialize() called twice: initial connect + after proactive reconnect
    assert session.initialize.call_count == 2


async def test_build_registry_wires_max_calls_from_config():
    """build_registry must forward max_calls_before_reconnect from config to MCPAdapter."""
    from kgent.agent.instrumentation import build_registry
    from kgent.config.loader import (
        AgentCoreConfig, KgentConfig, MCPConfig, MCPServerConfig,
        ModelConfig, ToolsConfig,
    )

    config = KgentConfig(
        model=ModelConfig(provider="local", base_url="http://x", api_key="x", model_name="m"),
        agent=AgentCoreConfig(),
        tools=ToolsConfig(static=[]),
        mcp=MCPConfig(servers=[
            MCPServerConfig(name="srv", command="cmd", args=[], max_calls_before_reconnect=77)
        ]),
    )

    captured: list[dict] = []

    class _FakeAdapter:
        def __init__(self, name, cmd, args, env, max_calls=None):
            captured.append({"name": name, "max_calls": max_calls})
            self._tools: list = []

        async def connect(self):
            pass

        def list_tools(self):
            return []

    with (
        patch("kgent.tools.mcp_adapter.MCP_AVAILABLE", True),
        patch("kgent.tools.mcp_adapter.MCPAdapter", _FakeAdapter),
    ):
        await build_registry(config)

    assert len(captured) == 1
    assert captured[0]["max_calls"] == 77
