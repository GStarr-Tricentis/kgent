from __future__ import annotations

import asyncio
import logging

import anyio
import anyio.abc

from kgent.agent.types import RegisteredTool, ToolSource

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class MCPAdapter:
    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None, max_calls: int | None = None) -> None:
        self._name = name
        self._command = command
        self._args = args
        self._env = env or {}
        self._lock = asyncio.Lock()
        self._call_count = 0
        self._max_calls: int | None = max_calls
        self._session = None
        self._tg = None              # anyio TaskGroup owning the lifecycle task
        self._disconnect_event = None  # signals the lifecycle task to stop
        self._connect_error: BaseException | None = None
        self._tools: list[RegisteredTool] = []
        self._session_ended: anyio.Event | None = None
        self._clean_disconnect: bool = False
        self._reconnect_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        return self._tg is not None

    async def connect(self) -> None:
        if self._tg is not None:
            return
        if not MCP_AVAILABLE:
            logger.warning("mcp package not installed, skipping server '%s'", self._name)
            return

        self._disconnect_event = anyio.Event()
        session_ready = anyio.Event()
        self._connect_error = None
        self._session_ended = anyio.Event()
        self._clean_disconnect = False

        # Start an anyio TaskGroup and run the session lifecycle as a subtask.
        # All of stdio_client's anyio cancel scopes live in that subtask's
        # context, keeping them isolated from the host task's cancel scope stack.
        self._tg = anyio.create_task_group()
        await self._tg.__aenter__()
        self._tg.start_soon(self._session_lifecycle, session_ready)

        # Wait until the lifecycle task signals readiness (or failure).
        await session_ready.wait()
        if self._connect_error is not None:
            exc = self._connect_error
            await self.disconnect()
            raise exc

    async def _session_lifecycle(self, session_ready: anyio.Event) -> None:
        """Run inside its own anyio task so cancel-scope cleanup is isolated."""
        try:
            params = StdioServerParameters(
                command=self._command, args=self._args, env=self._env or None
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    if not result.tools:
                        raise RuntimeError(
                            f"MCP server '{self._name}' connected but returned no tools"
                            " — check server config"
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
                    self._session = session
                    session_ready.set()
                    if self._disconnect_event is None:
                        raise RuntimeError("MCPAdapter._disconnect_event is None after connect")
                    await self._disconnect_event.wait()
        except anyio.get_cancelled_exc_class():
            raise
        except Exception as exc:
            self._connect_error = exc
            if not session_ready.is_set():
                session_ready.set()
        finally:
            if self._session_ended is not None and not self._session_ended.is_set():
                self._session_ended.set()

    def _make_callable(self, name: str):
        async def _call(args):
            return await self.call_tool(name, args)
        return _call

    async def disconnect(self) -> None:
        # Signal the lifecycle task to exit its blocking wait.
        if self._disconnect_event is not None:
            self._disconnect_event.set()

        # Close the task group — this cancels + awaits the lifecycle task.
        # stdio_client cleanup runs inside that task's cancel scope, not ours.
        if self._tg is not None:
            try:
                await self._tg.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug("MCP '%s': lifecycle cleanup error: %s", self._name, exc)
        self._session = None
        self._tg = None
        self._disconnect_event = None
        self._tools = []

    async def reconnect(self) -> None:
        async with self._reconnect_lock:
            await self.connect()

    async def shutdown(self) -> None:
        self._clean_disconnect = True
        await self.disconnect()

    async def monitor(self) -> bool:
        if self._session_ended is None:
            return False
        await self._session_ended.wait()
        return not self._clean_disconnect

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc_info):
        await self.shutdown()

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
                async with self._reconnect_lock:
                    await self.disconnect()
                    self._call_count = 0
                    await self.connect()

            result = await self._session.call_tool(name, arguments)
            self._call_count += 1
            return str(result.content)
