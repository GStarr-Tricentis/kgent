from __future__ import annotations

import asyncio
import contextlib
import logging

from agent_poc.agent.types import RegisteredTool, ToolSource

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class MCPAdapter:
    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None) -> None:
        self._name = name
        self._command = command
        self._args = args
        self._env = env or {}
        self._lock = asyncio.Lock()
        self._call_count = 0
        self._max_calls: int | None = None
        self._session = None
        self._stack = None
        self._tools: list[RegisteredTool] = []

    async def connect(self) -> None:
        if not MCP_AVAILABLE:
            logger.warning("mcp package not installed, skipping server '%s'", self._name)
            return

        params = StdioServerParameters(command=self._command, args=self._args, env=self._env or None)
        self._stack = contextlib.AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        result = await self._session.list_tools()
        if not result.tools:
            raise RuntimeError(
                f"MCP server '{self._name}' connected but returned no tools — check server config"
            )
        self._tools = [
            RegisteredTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {"type": "object", "properties": {}},
                callable=self._make_callable(t.name),
                source=ToolSource.MCP,
            )
            for t in result.tools
        ]
        logger.info("MCP '%s': registered %d tools", self._name, len(self._tools))

    def _make_callable(self, name: str):
        async def _call(args):
            return await self.call_tool(name, args)
        return _call

    async def disconnect(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._session = None
        self._stack = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc_info):
        await self.disconnect()

    def list_tools(self) -> list[RegisteredTool]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict) -> str:
        if not MCP_AVAILABLE:
            return "ERROR: mcp package not installed"

        async with self._lock:
            if self._max_calls and self._call_count >= self._max_calls:
                logger.info(
                    "MCP '%s': proactive reconnect after %d calls", self._name, self._call_count
                )
                await self._stack.aclose()
                self._call_count = 0
                await self.connect()
            try:
                result = await self._session.call_tool(name, arguments)
                self._call_count += 1
                return str(result.content)
            except (BrokenPipeError, ConnectionError, EOFError) as exc:
                logger.warning("MCP session error (%s), reconnecting", exc)
                await self._stack.aclose()
                self._call_count = 0
                await self.connect()
                result = await self._session.call_tool(name, arguments)
                self._call_count += 1
                return str(result.content)
