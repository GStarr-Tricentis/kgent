from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when launched via `streamlit run ui/app.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_poc.config.loader import load_config, load_dotenv

load_dotenv()

import streamlit as st

from agent_poc.agent.instrumentation import (
    TokenUsage,
    TrackingBackend,
    TimingRegistry,
    build_registry,
    reconstruct_timed_results,
)
from agent_poc.agent.runner import AgentRunner
from agent_poc.config.loader import AgentPocConfig
from agent_poc.models.openai_compatible import OpenAICompatibleBackend
from ui.components import TimedToolResult, render_run_summary, render_tool_card
from ui.config import CONFIG_PATH, get_ollama_models

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_YAML_PATH = "agent_poc/config/config.yaml"
SYSTEM_PROMPT_PATH = Path("agent_poc/prompts/system.txt")


def _build_registry(config: AgentPocConfig) -> TimingRegistry:
    return build_registry(config, warn_fn=st.warning)


def _reconstruct_timed_results(state, registry: TimingRegistry) -> list[TimedToolResult]:
    return reconstruct_timed_results(state, registry)


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
if "benchmark_rows" not in st.session_state:
    st.session_state.benchmark_rows = []  # list[dict]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

chat_tab, benchmark_tab = st.tabs(["Chat", "Benchmark"])

# ===========================================================================
# Chat tab
# ===========================================================================

with chat_tab:
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

    chat_col, tool_col = st.columns([7, 3])

    with chat_col:
        for msg in st.session_state.messages:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])

        prompt = st.chat_input("Send a message…")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

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

# ===========================================================================
# Benchmark tab
# ===========================================================================

with benchmark_tab:
    import csv
    import io

    import pandas as pd

    st.subheader("Benchmark")

    uploaded = st.file_uploader("Queries CSV", type="csv")
    bench_models = st.multiselect(
        "Models",
        options=get_ollama_models(),
        default=get_ollama_models()[:1],
    )
    reps = st.number_input("Repetitions per run", min_value=1, max_value=10, value=3)

    if st.button("Run Benchmark") and uploaded and bench_models:
        queries_df = pd.read_csv(uploaded)
        if "id" not in queries_df.columns:
            queries_df["id"] = range(len(queries_df))
        queries = queries_df.to_dict("records")

        config = load_config(CONFIG_YAML_PATH)
        system_prompt = (
            SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""
        )

        rows: list[dict] = []
        run_id = 0

        with st.spinner("Running benchmarks…"):
            for model in bench_models:
                config.model.model_name = model
                registry = _build_registry(config)

                for row in queries:
                    query_text = row["query"]
                    use_case = row.get("use_case", "")
                    query_id = row["id"]

                    for rep in range(1, int(reps) + 1):
                        run_id += 1
                        registry.reset()
                        usage = TokenUsage()
                        backend = TrackingBackend(
                            OpenAICompatibleBackend(config.model), usage
                        )
                        runner = AgentRunner(
                            backend=backend,
                            registry=registry,
                            config=config,
                            system_prompt=system_prompt,
                        )

                        error = False
                        wall_time = 0.0
                        state = None
                        try:
                            t0 = time.perf_counter()
                            state = runner.run(query_text)
                            wall_time = time.perf_counter() - t0
                        except Exception:
                            error = True

                        tool_names = []
                        tool_latencies = []
                        for result, elapsed_ms, _ in registry.timed_results:
                            tool_names.append(result.name)
                            tool_latencies.append(elapsed_ms)

                        mean_latency = (
                            sum(tool_latencies) / len(tool_latencies)
                            if tool_latencies
                            else ""
                        )

                        pt = usage.prompt_tokens or ""
                        ct = usage.completion_tokens or ""
                        tt = (usage.total_tokens) if (usage.prompt_tokens or usage.completion_tokens) else ""

                        rows.append({
                            "run_id": run_id,
                            "model": model,
                            "use_case": use_case,
                            "query_id": query_id,
                            "query": query_text,
                            "rep": rep,
                            "finish_reason": state.finish_reason if state else "",
                            "iterations": state.iteration if state else "",
                            "wall_time_s": round(wall_time, 3),
                            "prompt_tokens": pt,
                            "response_tokens": ct,
                            "total_tokens": tt,
                            "num_tool_calls": len(registry.timed_results),
                            "tool_names": ";".join(tool_names),
                            "tool_latencies_ms": ";".join(f"{l:.1f}" for l in tool_latencies),
                            "mean_tool_latency_ms": round(mean_latency, 1) if mean_latency != "" else "",
                            "response": _last_assistant_reply(state) if state else "",
                            "error": "true" if error else "false",
                        })

        st.session_state.benchmark_rows = rows

    if st.session_state.benchmark_rows:
        df = pd.DataFrame(st.session_state.benchmark_rows)
        st.dataframe(df)

        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=list(st.session_state.benchmark_rows[0].keys()),
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(st.session_state.benchmark_rows)
        st.download_button(
            "Download CSV",
            data=buf.getvalue(),
            file_name="benchmark_results.csv",
            mime="text/csv",
        )
