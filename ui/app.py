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
from agent_poc.models.factory import make_backend
from ui.components import TimedToolResult, render_run_summary, render_tool_card
from ui.config import CONFIG_PATH, get_ollama_models

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_YAML_PATH = "agent_poc/config/config.yaml"
_PROMPT_DIR = Path("agent_poc/agent/prompts")

TRICENTIS_DEPLOYMENTS = [
    "anthropic.claude-sonnet-4-6",
    "anthropic.claude-opus-4-6-v1",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-opus-4-5-20251101-v1:0",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-1106-Preview",
    "gpt-35-turbo-0613",
    "gpt-35-turbo-0301",
]


def _build_registry(
    config: AgentPocConfig,
    skip_neo4j: bool = False,
) -> TimingRegistry:
    skip = frozenset({"neo4j"}) if skip_neo4j else frozenset()
    return build_registry(config, warn_fn=st.warning, skip_servers=skip)


def _reconstruct_timed_results(state, registry: TimingRegistry) -> list[TimedToolResult]:
    return reconstruct_timed_results(state, registry)


def _last_assistant_reply(state) -> str:
    for msg in reversed(state.messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return "[Agent stopped without a text response]"


def _system_prompt(cypher_tool_enabled: bool) -> str:
    base_path = Path("agent_poc/prompts/system.txt")
    system_prompt = base_path.read_text() if base_path.exists() else ""
    if cypher_tool_enabled:
        graph_path = _PROMPT_DIR / "text_to_cypher_tool.txt"
        if graph_path.exists():
            graph_prompt = graph_path.read_text()
            system_prompt = system_prompt + ("\n\n" if system_prompt else "") + graph_prompt
    return system_prompt


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Open-weight Agent", layout="wide")
st.title("Open-weight Agent")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_timed" not in st.session_state:
    st.session_state.last_timed = []
if "last_run_state" not in st.session_state:
    st.session_state.last_run_state = None
if "last_usage" not in st.session_state:
    st.session_state.last_usage = None
if "last_elapsed" not in st.session_state:
    st.session_state.last_elapsed = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "benchmark_rows" not in st.session_state:
    st.session_state.benchmark_rows = []
if "provider" not in st.session_state:
    st.session_state.provider = "local"
if "tricentis_deployment" not in st.session_state:
    st.session_state.tricentis_deployment = ""
if "cypher_tool_enabled" not in st.session_state:
    st.session_state.cypher_tool_enabled = False

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

    top_cols = st.columns([2, 3, 2, 1])

    with top_cols[0]:
        st.session_state.provider = st.radio(
            "Provider",
            ["local", "tricentis"],
            index=["local", "tricentis"].index(st.session_state.provider),
            horizontal=True,
            label_visibility="collapsed",
        )

    with top_cols[1]:
        if st.session_state.provider == "local":
            if st.session_state.selected_model not in models:
                st.session_state.selected_model = models[0] if models else ""
            st.session_state.selected_model = st.selectbox(
                "Model",
                options=models,
                index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0,
                label_visibility="collapsed",
            )
        else:
            if st.session_state.tricentis_deployment not in TRICENTIS_DEPLOYMENTS:
                st.session_state.tricentis_deployment = TRICENTIS_DEPLOYMENTS[0]
            st.session_state.tricentis_deployment = st.selectbox(
                "Deployment",
                options=TRICENTIS_DEPLOYMENTS,
                index=TRICENTIS_DEPLOYMENTS.index(st.session_state.tricentis_deployment),
                label_visibility="collapsed",
            )

    with top_cols[2]:
        st.session_state.cypher_tool_enabled = st.checkbox(
            "Cypher tool", value=st.session_state.cypher_tool_enabled
        )

    with top_cols[3]:
        if st.session_state.last_run_state is not None:
            st.caption(f"Status: {st.session_state.last_run_state.finish_reason}")

    if st.session_state.provider == "tricentis":
        st.info(
            "If authentication is required, an SSO link will appear in the terminal "
            "where you launched Streamlit. Complete sign-in there, then retry your message.",
            icon="ℹ️",
        )

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

            cypher_on = st.session_state.cypher_tool_enabled
            provider = st.session_state.provider

            if provider == "local":
                model_override = st.session_state.selected_model
            else:
                model_override = st.session_state.tricentis_deployment or None

            system_prompt = _system_prompt(cypher_on)
            registry = _build_registry(config, skip_neo4j=cypher_on)

            if cypher_on:
                from agent_poc.tools.cypher_tool import make_cypher_tool
                registry.register(make_cypher_tool(config))

            usage = TokenUsage()
            backend = TrackingBackend(
                make_backend(config, provider=provider, model_override=model_override),
                usage,
            )
            runner = AgentRunner(
                backend=backend,
                registry=registry,
                config=config,
                system_prompt=system_prompt,
            )

            spinner_msg = (
                "Agent is thinking… (if paused, check terminal for SSO prompt)"
                if provider == "tricentis"
                else "Agent is thinking…"
            )
            with st.spinner(spinner_msg):
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

    bench_provider = st.radio("Provider", ["local", "tricentis"], horizontal=True, key="bench_provider")

    if bench_provider == "local":
        bench_models = st.multiselect(
            "Models",
            options=get_ollama_models(),
            default=get_ollama_models()[:1],
        )
        bench_deployment = ""
    else:
        bench_models = []
        bench_deployment = st.selectbox(
            "Deployment",
            options=TRICENTIS_DEPLOYMENTS,
            key="bench_deployment",
        )

    bench_cypher_tool = st.checkbox("Cypher tool", key="bench_cypher_tool")
    reps = st.number_input("Repetitions per run", min_value=1, max_value=10, value=3)

    run_ready = uploaded and (bench_models if bench_provider == "local" else bench_deployment)

    if st.button("Run Benchmark") and run_ready:
        queries_df = pd.read_csv(uploaded)
        if "id" not in queries_df.columns:
            queries_df["id"] = range(len(queries_df))
        queries = queries_df.to_dict("records")

        config = load_config(CONFIG_YAML_PATH)
        system_prompt = _system_prompt(bench_cypher_tool)

        rows: list[dict] = []
        run_id = 0

        # For local: iterate over selected models. For tricentis: single "model" = deployment.
        model_list = bench_models if bench_provider == "local" else [bench_deployment]

        with st.spinner("Running benchmarks…"):
            for model in model_list:
                if bench_provider == "local":
                    config.model.model_name = model
                    model_override = model
                else:
                    model_override = model  # deployment string

                registry = _build_registry(config, skip_neo4j=bench_cypher_tool)

                if bench_cypher_tool:
                    from agent_poc.tools.cypher_tool import make_cypher_tool
                    registry.register(make_cypher_tool(config))

                for row in queries:
                    query_text = row["query"]
                    use_case = row.get("use_case", "")
                    query_id = row["id"]

                    for rep in range(1, int(reps) + 1):
                        run_id += 1
                        registry.reset()
                        usage = TokenUsage()
                        backend = TrackingBackend(
                            make_backend(config, provider=bench_provider, model_override=model_override),
                            usage,
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
                            "provider": bench_provider,
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
