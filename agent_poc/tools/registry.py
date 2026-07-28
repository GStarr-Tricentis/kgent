from __future__ import annotations

import asyncio
import inspect
import logging

from agent_poc.agent.types import RegisteredTool, ToolCall, ToolResult

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
            if inspect.iscoroutinefunction(tool.callable):
                coro = tool.callable(call.arguments)
            else:
                coro = asyncio.to_thread(tool.callable, call.arguments)
            output = await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                error=True,
                output=f"Tool '{call.name}' timed out after {timeout}s.",
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                error=True,
                output=f"Tool '{call.name}' raised an exception: {exc}",
            )

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            output=str(output),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        for adapter in self._adapters:
            await adapter.disconnect()
