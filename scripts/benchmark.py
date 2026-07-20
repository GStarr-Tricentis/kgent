#!/usr/bin/env python3
"""CLI benchmark runner: runs a set of queries from a CSV across multiple models."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_poc.config.loader import load_config, load_dotenv

load_dotenv()

from agent_poc.agent.instrumentation import TokenUsage, TrackingBackend, build_registry
from agent_poc.agent.runner import AgentRunner

SYSTEM_PROMPT_PATH = Path("agent_poc/prompts/system.txt")

OUTPUT_FIELDS = [
    "run_id", "model", "use_case", "query_id", "query", "rep",
    "graph_mode",
    "finish_reason", "iterations", "wall_time_s",
    "prompt_tokens", "response_tokens", "total_tokens",
    "num_tool_calls", "tool_names", "tool_latencies_ms", "mean_tool_latency_ms",
    "response", "error", "provider",
]


def _last_assistant_reply(state) -> str:
    for msg in reversed(state.messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


def _read_queries(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        if "id" not in row or not row["id"]:
            row["id"] = str(i)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark open-weight agent across models and queries")
    parser.add_argument("--queries", required=True, help="Path to input CSV")
    parser.add_argument(
        "--models", required=True,
        help=(
            "Comma-separated model names. For 'local' these are Ollama model tags "
            "(e.g. qwen3:8b). For 'tricentis' these are deployment names "
            "(e.g. anthropic.claude-sonnet-4-6). For 'bedrock' these are model IDs "
            "(e.g. qwen.qwen3-coder-next)."
        ),
    )
    parser.add_argument("--output", default="benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per query×model")
    parser.add_argument("--config", default="agent_poc/config/config.yaml", help="Agent config path")
    parser.add_argument("--provider", default="local", choices=["local", "tricentis", "bedrock"],
                        help="Model provider (default: local)")
    parser.add_argument("--cypher-tool", action="store_true",
                        help="Enable the NLP-to-Cypher tool (disables the neo4j MCP server)")
    parser.add_argument("--no-graph", action="store_true",
                        help="Disable all graph access (no Neo4j MCP, no Cypher tool) for a straight-RAG baseline")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    queries = _read_queries(Path(args.queries))
    config = load_config(args.config)
    system_prompt = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""
    if args.cypher_tool:
        graph_path = Path("agent_poc/agent/prompts/text_to_cypher_tool.txt")
        if graph_path.exists():
            system_prompt = system_prompt + ("\n\n" if system_prompt else "") + graph_path.read_text()

    if args.no_graph and args.cypher_tool:
        parser.error("--no-graph and --cypher-tool are mutually exclusive")

    if args.no_graph:
        graph_mode = "none"
    elif args.cypher_tool:
        graph_mode = "cypher_tool"
    else:
        graph_mode = "neo4j_mcp"

    total_runs = len(models) * len(queries) * args.reps

    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        run_id = 0
        for model in models:
            config.model.model_name = model
            if graph_mode == "none":
                skip = frozenset({"neo4j"})
            elif graph_mode == "cypher_tool":
                skip = frozenset({"neo4j"})
            else:
                skip = frozenset()
            registry = build_registry(config, skip_servers=skip)
            if graph_mode == "cypher_tool":
                from agent_poc.tools.cypher_tool import make_cypher_tool
                registry.register(make_cypher_tool(config))

            for row in queries:
                query_text = row["query"]
                use_case = row.get("use_case", "")
                query_id = row["id"]

                for rep in range(1, args.reps + 1):
                    run_id += 1
                    registry.reset()
                    usage = TokenUsage()
                    from agent_poc.models.factory import make_backend
                    backend = TrackingBackend(make_backend(config, provider=args.provider, model_override=model), usage)
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
                        t0 = perf_counter()
                        state = runner.run(query_text)
                        wall_time = perf_counter() - t0
                    except Exception as exc:
                        error = True
                        print(f"[error] run {run_id}: {exc}", file=sys.stderr)

                    tool_names = []
                    tool_latencies = []
                    for result, elapsed_ms, _ in registry.timed_results:
                        tool_names.append(result.name)
                        tool_latencies.append(elapsed_ms)

                    mean_latency = (
                        sum(tool_latencies) / len(tool_latencies) if tool_latencies else ""
                    )

                    pt = usage.prompt_tokens or ""
                    ct = usage.completion_tokens or ""
                    tt = usage.total_tokens if (usage.prompt_tokens or usage.completion_tokens) else ""

                    writer.writerow({
                        "run_id": run_id,
                        "model": model,
                        "use_case": use_case,
                        "graph_mode": graph_mode,
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
                        "provider": args.provider,
                    })
                    out_f.flush()

                    preview = query_text[:40].replace("\n", " ")
                    done = f"{wall_time:.1f}s" if not error else "error"
                    print(
                        f"[model {model} | query {queries.index(row) + 1}/{len(queries)} \"{preview}\" | rep {rep}/{args.reps}] {done}",
                        file=sys.stderr,
                    )

    print(f"Done. {run_id}/{total_runs} runs written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
