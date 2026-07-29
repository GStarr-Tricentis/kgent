"""scripts/query.py — Natural language querying against the Neo4j knowledge graph.

Usage:
    python scripts/query.py --question "How many test cases are in the graph?"
                            [--model qwen2.5-coder:14b] [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Neo4j knowledge graph in natural language")
    parser.add_argument("--question", required=True, help="Natural language question to answer")
    parser.add_argument("--model", default=None, help="Override model from config")
    parser.add_argument("--config", default="kgent/config/config.yaml")
    parser.add_argument("--provider", default="local", choices=["local", "tricentis"],
                        help="Model provider (default: local)")
    parser.add_argument("--cypher-tool", action="store_true",
                        help="Use query_graph tool instead of raw Neo4j MCP tools")
    args = parser.parse_args()

    from kgent.config.loader import load_config, load_dotenv
    load_dotenv()
    config = load_config(args.config)
    if args.model:
        config.model.model_name = args.model

    prompt_file = "text_to_cypher_tool.txt" if args.cypher_tool else "text_to_cypher.txt"
    prompt_path = Path("kgent/agent/prompts") / prompt_file
    if not prompt_path.exists():
        print(f"ERROR: system prompt not found at {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text()

    from kgent.agent.instrumentation import build_registry
    from kgent.agent.runner import AgentRunner
    from kgent.models.factory import make_backend

    skip_servers = frozenset({"neo4j"}) if args.cypher_tool else frozenset()
    registry = await build_registry(config, skip_servers=skip_servers)

    async with registry:
        if args.cypher_tool:
            from kgent.tools.cypher_tool import make_cypher_tool
            registry.register(await make_cypher_tool(config))

        backend = await make_backend(config, provider=args.provider, model_override=args.model)
        runner = AgentRunner(backend=backend, registry=registry, config=config, system_prompt=system_prompt)
        state = await runner.run(args.question)

        last_content = next(
            (m["content"] for m in reversed(state.messages) if m.get("role") == "assistant" and m.get("content")),
            None,
        )
        if last_content:
            print(last_content)
        else:
            print(f"(Agent finished with reason: {state.finish_reason})")


if __name__ == "__main__":
    asyncio.run(main())
