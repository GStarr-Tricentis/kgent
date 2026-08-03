from __future__ import annotations

import asyncio
import inspect
import logging

import anyio

from kgent.agent.types import RegisteredTool, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._adapters: list = []

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s (source=%s)", tool.name, tool.source)

    def register_adapter(self, adapter) -> None:
        self._adapters.append(adapter)

    @property
    def adapters(self):
        return list(self._adapters)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    async def execute(self, call: ToolCall, timeout_override: float | None = None) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                error=True,
                output=(
                    f"Unknown tool '{call.name}'. "
                    f"Available tools: {list(self._tools.keys())}"
                ),
            )

        timeout = timeout_override if timeout_override is not None else tool.timeout_seconds

        try:
            with anyio.move_on_after(timeout) as cancel_scope:
                if inspect.iscoroutinefunction(tool.callable):
                    output = await tool.callable(call.arguments)
                else:
                    output = await asyncio.to_thread(tool.callable, call.arguments)
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                error=True,
                output=f"Tool '{call.name}' raised an exception: {exc}",
            )

        if cancel_scope.cancelled_caught:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                error=True,
                output=f"Tool '{call.name}' timed out after {timeout}s.",
            )

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            output=str(output),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        for tool in self._tools.values():
            if tool.close is not None:
                try:
                    result = tool.close()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    logger.debug("Tool '%s' close error: %s", tool.name, exc)
        for adapter in self._adapters:
            try:
                await adapter.shutdown()
            except Exception as exc:
                logger.debug("Adapter shutdown error: %s", exc)
