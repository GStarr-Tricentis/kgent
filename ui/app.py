from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path when launched via `streamlit run ui/app.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_poc.config.loader import load_config, load_dotenv

load_dotenv()

import streamlit as st

from agent_poc.agent.runner import AgentRunner
from agent_poc.agent.types import ModelBackend, ModelResponse, RegisteredTool, ToolCall, ToolResult, ToolSource
from agent_poc.config.loader import AgentPocConfig
from agent_poc.models.openai_compatible import OpenAICompatibleBackend
from agent_poc.tools.generated import make_save_as_tool
from agent_poc.tools.mcp_adapter import MCP_AVAILABLE, MCPAdapter
from agent_poc.tools.registry import ToolRegistry
from agent_poc.tools.static.filesystem import LIST_DIR_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL
from agent_poc.tools.static.python_exec import make_python_exec_tool
from agent_poc.tools.static.shell import RUN_COMMAND_TOOL
from ui.components import TimedToolResult, render_run_summary, render_tool_card
from ui.config import CONFIG_PATH, get_ollama_models

# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

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
        # each entry: (result, elapsed_ms, arguments)

    def execute(self, call: ToolCall, timeout_override: float | None = None) -> ToolResult:
        start = time.perf_counter()
        result = super().execute(call, timeout_override)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.timed_results.append((result, elapsed_ms, call.arguments))
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_YAML_PATH = "agent_poc/config/config.yaml"
SYSTEM_PROMPT_PATH = Path("agent_poc/prompts/system.txt")


def _build_registry(config: AgentPocConfig) -> TimingRegistry:
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
                st.warning(f"[mcp] {srv.name} failed: {exc}")
    return registry


def _reconstruct_timed_results(
    state,
    registry: TimingRegistry,
) -> list[TimedToolResult]:
    """
    Match each (result, elapsed_ms, args) in registry.timed_results to the
    iteration it occurred in, using tool_call_id from state.messages.
    Returns a list of TimedToolResult in execution order.
    """
    # Build a map from tool_call_id → iteration number by walking messages.
    # Each assistant message with tool_calls marks the start of that iteration's calls.
    tool_id_to_iteration: dict[str, int] = {}
    iteration = 0
    for msg in state.messages:
        if msg.get("role") == "assistant":
            calls = msg.get("tool_calls") or []
            if calls:
                iteration += 1
                for tc in calls:
                    # OpenAI SDK format: tc is a dict with 'id' at top level
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


def _last_assistant_reply(state) -> str:
    for msg in reversed(state.messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return "[Agent stopped without a text response]"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Open-weight Agent", layout="wide")
st.title("Open-weight Agent")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict] — display history
if "last_timed" not in st.session_state:
    st.session_state.last_timed = []  # list[TimedToolResult]
if "last_run_state" not in st.session_state:
    st.session_state.last_run_state = None
if "last_usage" not in st.session_state:
    st.session_state.last_usage = None  # TokenUsage | None
if "last_elapsed" not in st.session_state:
    st.session_state.last_elapsed = None  # float | None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# ---------------------------------------------------------------------------
# Top bar — model selector + status
# ---------------------------------------------------------------------------

models = get_ollama_models()
if st.session_state.selected_model not in models:
    st.session_state.selected_model = models[0] if models else ""

top_cols = st.columns([3, 1])
with top_cols[0]:
    st.session_state.selected_model = st.selectbox(
        "Model",
        options=models,
        index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0,
        label_visibility="collapsed",
    )
with top_cols[1]:
    if st.session_state.last_run_state is not None:
        reason = st.session_state.last_run_state.finish_reason
        st.caption(f"Status: {reason}")

# ---------------------------------------------------------------------------
# Main layout: chat (70%) | tool calls (30%)
# ---------------------------------------------------------------------------

chat_col, tool_col = st.columns([7, 3])

# --- Chat column ---
with chat_col:
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

    prompt = st.chat_input("Send a message…")

    if prompt:
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Reset previous run data
        st.session_state.last_timed = []
        st.session_state.last_run_state = None
        st.session_state.last_usage = None
        st.session_state.last_elapsed = None

        config = load_config(CONFIG_YAML_PATH)
        config.model.model_name = st.session_state.selected_model

        system_prompt = (
            SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""
        )

        registry = _build_registry(config)
        usage = TokenUsage()
        backend = TrackingBackend(OpenAICompatibleBackend(config.model), usage)
        runner = AgentRunner(
            backend=backend,
            registry=registry,
            config=config,
            system_prompt=system_prompt,
        )

        with st.spinner("Agent is thinking…"):
            t0 = time.perf_counter()
            state = runner.run(prompt)
            elapsed = time.perf_counter() - t0

        reply = _last_assistant_reply(state)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.last_timed = _reconstruct_timed_results(state, registry)
        st.session_state.last_run_state = state
        st.session_state.last_usage = usage
        st.session_state.last_elapsed = elapsed

        st.rerun()

# --- Tool call column ---
with tool_col:
    st.subheader("Tool Calls")

    if not st.session_state.last_timed:
        st.caption("Tool calls will appear here after a run.")
    else:
        for i, ttr in enumerate(st.session_state.last_timed):
            render_tool_card(ttr, i)

        state = st.session_state.last_run_state
        usage = st.session_state.last_usage
        elapsed = st.session_state.last_elapsed

        render_run_summary(
            iteration=state.iteration,
            finish_reason=state.finish_reason,
            total_elapsed=elapsed,
            prompt_tokens=usage.prompt_tokens if usage and usage.prompt_tokens else None,
            response_tokens=usage.completion_tokens if usage and usage.completion_tokens else None,
        )
