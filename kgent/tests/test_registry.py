from __future__ import annotations

import asyncio

import pytest

from kgent.agent.types import RegisteredTool, ToolCall, ToolSource
from kgent.tools.registry import ToolRegistry


def _tool(name: str, fn, timeout: float = 30.0, close=None) -> RegisteredTool:
    return RegisteredTool(
        name=name,
        description="",
        input_schema={"type": "object", "properties": {}},
        callable=fn,
        source=ToolSource.STATIC,
        timeout_seconds=timeout,
        close=close,
    )


def _call(name: str, args: dict = {}, call_id: str = "tc1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args)


# --- unknown tool ---

async def test_unknown_tool():
    r = ToolRegistry()
    result = await r.execute(_call("ghost"))
    assert result.error is True
    assert "ghost" in result.output


async def test_unknown_tool_lists_available():
    r = ToolRegistry()
    r.register(_tool("real_tool", lambda args: "ok"))
    result = await r.execute(_call("ghost"))
    assert "real_tool" in result.output


# --- successful execution ---

async def test_successful_execution():
    r = ToolRegistry()
    r.register(_tool("greet", lambda args: "hello"))
    result = await r.execute(_call("greet"))
    assert result.output == "hello"
    assert result.error is False
    assert result.tool_call_id == "tc1"
    assert result.name == "greet"


async def test_successful_execution_passes_args():
    r = ToolRegistry()
    r.register(_tool("echo", lambda args: args.get("text", "")))
    result = await r.execute(_call("echo", {"text": "world"}))
    assert result.output == "world"


async def test_async_callable_is_dispatched_directly():
    """Async tool callables bypass asyncio.to_thread and are awaited directly."""
    r = ToolRegistry()

    async def _async_greet(args):
        return "async hello"

    r.register(_tool("async_greet", _async_greet))
    result = await r.execute(_call("async_greet"))
    assert result.output == "async hello"
    assert result.error is False


# --- timeout ---

async def test_timeout():
    r = ToolRegistry()

    async def _slow(args):
        await asyncio.sleep(60)

    r.register(_tool("slow", _slow, timeout=0.1))
    result = await r.execute(_call("slow"))
    assert result.error is True
    assert "timeout" in result.output.lower() or "timed out" in result.output.lower()


async def test_timeout_does_not_raise():
    r = ToolRegistry()

    async def _slow(args):
        await asyncio.sleep(60)

    r.register(_tool("slow", _slow, timeout=0.1))
    result = await r.execute(_call("slow"))
    assert isinstance(result.output, str)


# --- exception in callable ---

async def test_exception_in_callable():
    r = ToolRegistry()
    r.register(_tool("bad", lambda args: (_ for _ in ()).throw(RuntimeError("boom"))))
    result = await r.execute(_call("bad"))
    assert result.error is True
    assert "boom" in result.output


async def test_exception_in_callable_does_not_raise():
    r = ToolRegistry()
    r.register(_tool("bad", lambda args: 1 / 0))
    result = await r.execute(_call("bad"))
    assert result.error is True


# --- timeout_override ---

async def test_timeout_override_shortens_timeout():
    r = ToolRegistry()

    async def _slow(args):
        await asyncio.sleep(60)

    r.register(_tool("slow", _slow, timeout=30.0))
    result = await r.execute(_call("slow"), timeout_override=0.1)
    assert result.error is True
    assert "timeout" in result.output.lower() or "timed out" in result.output.lower()


# --- adapter lifecycle ---

async def test_context_manager_disconnects_adapters_on_exit():
    from unittest.mock import AsyncMock

    r = ToolRegistry()
    adapter1 = AsyncMock()
    adapter2 = AsyncMock()
    r.register_adapter(adapter1)
    r.register_adapter(adapter2)

    async with r:
        pass

    adapter1.shutdown.assert_awaited_once()
    adapter2.shutdown.assert_awaited_once()


# --- close hook lifecycle ---

async def test_sync_close_hook_called_on_exit():
    closed = []
    r = ToolRegistry()
    r.register(_tool("t", lambda args: "ok", close=lambda: closed.append(1)))
    async with r:
        pass
    assert closed == [1]


async def test_async_close_hook_called_on_exit():
    closed = []

    async def _aclose():
        closed.append(1)

    r = ToolRegistry()
    r.register(_tool("t", lambda args: "ok", close=_aclose))
    async with r:
        pass
    assert closed == [1]


async def test_close_hook_called_before_adapters():
    from unittest.mock import AsyncMock
    call_log = []

    adapter = AsyncMock()
    async def _adapter_shutdown():
        call_log.append("adapter")
    adapter.shutdown.side_effect = _adapter_shutdown

    r = ToolRegistry()
    r.register(_tool("t", lambda args: "ok", close=lambda: call_log.append("tool")))
    r.register_adapter(adapter)
    async with r:
        pass
    assert call_log == ["tool", "adapter"]


async def test_close_hook_error_does_not_prevent_adapter_shutdown():
    from unittest.mock import AsyncMock

    def _bad_close():
        raise RuntimeError("close failed")

    r = ToolRegistry()
    r.register(_tool("t", lambda args: "ok", close=_bad_close))
    adapter = AsyncMock()
    r.register_adapter(adapter)
    async with r:
        pass
    adapter.shutdown.assert_awaited_once()
