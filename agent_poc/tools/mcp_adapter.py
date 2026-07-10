from __future__ import annotations

import asyncio
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
    # PoC limitation: reconnect-per-call. Each tool invocation opens a new
    # subprocess, performs the MCP handshake, calls the tool, then closes.
    # This avoids persistent session management at the cost of per-call latency.

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None) -> None:
        self._name = name
        self._command = command
        self._args = args
        self._env = env or {}
        self._tools: list[RegisteredTool] = []

    def connect(self) -> None:
        if not MCP_AVAILABLE:
            logger.warning("mcp package not installed, skipping server '%s'", self._name)
            return

        async def _fetch():
            params = StdioServerParameters(command=self._command, args=self._args, env=self._env or None)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return result.tools

        mcp_tools = asyncio.run(_fetch())

        # name=t.name default-argument capture is CRITICAL — without it every
        # lambda would close over the loop variable and call the last tool name.
        self._tools = [
            RegisteredTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {"type": "object", "properties": {}},
                callable=lambda args, name=t.name: self.call_tool(name, args),
                source=ToolSource.MCP,
            )
            for t in mcp_tools
        ]
        logger.info("MCP '%s': registered %d tools", self._name, len(self._tools))

    def list_tools(self) -> list[RegisteredTool]:
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict) -> str:
        if not MCP_AVAILABLE:
            return "ERROR: mcp package not installed"

        async def _call():
            params = StdioServerParameters(command=self._command, args=self._args, env=self._env or None)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    return str(result.content)

        return asyncio.run(_call())

    def disconnect(self) -> None:
        pass  # reconnect-per-call; nothing persistent to close
