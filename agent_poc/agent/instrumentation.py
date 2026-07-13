from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent_poc.agent.types import ModelBackend, ModelResponse, RegisteredTool, ToolCall, ToolResult, ToolSource
from agent_poc.config.loader import AgentPocConfig
from agent_poc.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TrackingBackend:
    """Wraps a ModelBackend and accumulates token usage across all calls."""

    def __init__(self, backend: ModelBackend, usage: TokenUsage) -> None:
        self._backend = backend
        self._usage = usage

    def complete(
        self, messages: list[dict], tools: list[RegisteredTool]
    ) -> ModelResponse:
        response = self._backend.complete(messages, tools)
        if response.raw and hasattr(response.raw, "usage") and response.raw.usage:
            self._usage.prompt_tokens += response.raw.usage.prompt_tokens or 0
            self._usage.completion_tokens += response.raw.usage.completion_tokens or 0
        return response


class TimingRegistry(ToolRegistry):
    """ToolRegistry that records elapsed time for every execute() call."""

    def __init__(self) -> None:
        super().__init__()
        self.timed_results: list[tuple[ToolResult, float, dict]] = []

    def reset(self) -> None:
        """Clear timed_results so the registry can be reused across benchmark runs."""
        self.timed_results = []

    def execute(self, call: ToolCall, timeout_override: float | None = None) -> ToolResult:
        start = time.perf_counter()
        result = super().execute(call, timeout_override)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.timed_results.append((result, elapsed_ms, call.arguments))
        return result


@dataclass
class TimedToolResult:
    result: ToolResult
    elapsed_ms: float
    iteration: int
    source: ToolSource
    args: dict = field(default_factory=dict)


def reconstruct_timed_results(state, registry: TimingRegistry) -> list[TimedToolResult]:
    """Match each timed registry entry to the iteration it occurred in."""
    tool_id_to_iteration: dict[str, int] = {}
    iteration = 0
    for msg in state.messages:
        if msg.get("role") == "assistant":
            calls = msg.get("tool_calls") or []
            if calls:
                iteration += 1
                for tc in calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id:
                        tool_id_to_iteration[tc_id] = iteration

    results: list[TimedToolResult] = []
    for result, elapsed_ms, args in registry.timed_results:
        iter_num = tool_id_to_iteration.get(result.tool_call_id, 0)
        tool = registry.get(result.name)
        source = tool.source if tool else ToolSource.STATIC
        results.append(
            TimedToolResult(
                result=result,
                elapsed_ms=elapsed_ms,
                iteration=iter_num,
                source=source,
                args=args,
            )
        )
    return results


def build_registry(config: AgentPocConfig, warn_fn=None) -> TimingRegistry:
    """
    Build a TimingRegistry from config. MCP connection errors are passed to
    warn_fn(message) if provided, otherwise logged to stderr.
    """
    from agent_poc.tools.generated import make_save_as_tool
    from agent_poc.tools.mcp_adapter import MCP_AVAILABLE, MCPAdapter
    from agent_poc.tools.static.filesystem import LIST_DIR_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL
    from agent_poc.tools.static.python_exec import make_python_exec_tool
    from agent_poc.tools.static.shell import RUN_COMMAND_TOOL

    registry = TimingRegistry()
    static_map = {
        "filesystem": [READ_FILE_TOOL, WRITE_FILE_TOOL, LIST_DIR_TOOL],
        "shell": [RUN_COMMAND_TOOL],
        "python_exec": [make_python_exec_tool(config.sandbox)],
    }
    for name in config.tools.static:
        for tool in static_map.get(name, []):
            registry.register(tool)
    registry.register(make_save_as_tool(registry, config.sandbox))

    if MCP_AVAILABLE:
        for srv in config.mcp.servers:
            try:
                adapter = MCPAdapter(srv.name, srv.command, srv.args, srv.expanded_env())
                adapter.connect()
                for t in adapter.list_tools():
                    registry.register(t)
            except Exception as exc:
                msg = f"[mcp] {srv.name} failed: {exc}"
                if warn_fn is not None:
                    warn_fn(msg)
                else:
                    print(msg, file=sys.stderr)

    return registry
