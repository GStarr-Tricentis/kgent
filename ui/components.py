from __future__ import annotations

import json

import streamlit as st

from agent_poc.agent.instrumentation import TimedToolResult
from agent_poc.agent.types import ToolResult, ToolSource

TRUNCATE_CHARS = 300


def _status_icon(result: ToolResult) -> tuple[str, str]:
    if "Repeated identical" in result.output:
        return "⚠", "orange"
    if result.error:
        return "✗", "red"
    return "✓", "green"


def render_tool_card(ttr: TimedToolResult, card_index: int) -> None:
    icon, color = _status_icon(ttr.result)

    badge = ""
    if ttr.source == ToolSource.GENERATED:
        badge = " 🔧 generated"
    elif ttr.source == ToolSource.MCP:
        badge = " 🔌 mcp"

    header = (
        f"[iter {ttr.iteration}] **{ttr.result.name}**{badge}"
        f" :{color}[{icon}] {ttr.elapsed_ms:.0f}ms"
    )

    with st.expander(header, expanded=True):
        st.code(json.dumps(ttr.args, indent=2, ensure_ascii=False), language="json")

        if ttr.result.name == "save_as_tool":
            name = ttr.args.get("name", "?")
            desc = ttr.args.get("description", "")
            st.info(f"Created tool **{name}**: {desc}")

        output = ttr.result.output
        if len(output) <= TRUNCATE_CHARS:
            st.text(output)
        else:
            st.text(output[:TRUNCATE_CHARS] + "…")
            with st.expander("show more"):
                st.text(output)


def render_run_summary(
    iteration: int,
    finish_reason: str,
    total_elapsed: float,
    prompt_tokens: int | None,
    response_tokens: int | None,
) -> None:
    st.markdown("---")
    st.markdown("**Run summary**")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Iterations", iteration)
        st.metric("Finish reason", finish_reason)
    with cols[1]:
        st.metric("Wall time", f"{total_elapsed:.2f}s")
        if prompt_tokens is not None:
            st.metric("Prompt tokens", f"{prompt_tokens:,}")
            st.metric("Response tokens", f"{response_tokens:,}")
            st.metric("Total tokens", f"{(prompt_tokens + response_tokens):,}")
        else:
            st.caption("Token counts unavailable for this model.")
