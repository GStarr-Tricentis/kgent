"""scripts/query.py — Natural language querying against the Neo4j knowledge graph.

Usage:
    python scripts/query.py --question "How many test cases are in the graph?"
                            [--model qwen2.5-coder:14b] [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Neo4j knowledge graph in natural language")
    parser.add_argument("--question", required=True, help="Natural language question to answer")
    parser.add_argument("--model", default=None, help="Override model from config")
    parser.add_argument("--config", default="agent_poc/config/config.yaml")
    parser.add_argument("--provider", default="local", choices=["local", "tricentis"],
                        help="Model provider (default: local)")
    args = parser.parse_args()

    from agent_poc.config.loader import load_config, load_dotenv
    load_dotenv()
    config = load_config(args.config)
    if args.model:
        config.model.model_name = args.model

    prompt_path = Path("agent_poc/agent/prompts/text_to_cypher.txt")
    if not prompt_path.exists():
        print(f"ERROR: system prompt not found at {prompt_path}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_path.read_text()

    from agent_poc.tools.registry import ToolRegistry
    from agent_poc.tools.mcp_adapter import MCP_AVAILABLE, MCPAdapter
    from agent_poc.agent.runner import AgentRunner

    registry = ToolRegistry()

    if MCP_AVAILABLE:
        for srv in config.mcp.servers:
            try:
                adapter = MCPAdapter(srv.name, srv.command, srv.args, srv.expanded_env())
                adapter.connect()
                for tool in adapter.list_tools():
                    registry.register(tool)
            except Exception as e:
                print(f"[mcp] {srv.name} failed to connect: {e}", file=sys.stderr)
    elif config.mcp.servers:
        print("[mcp] Warning: mcp package not installed — Neo4j tools unavailable", file=sys.stderr)

    from agent_poc.models.factory import make_backend
    backend = make_backend(config, provider=args.provider, model_override=args.model)
    runner = AgentRunner(backend=backend, registry=registry, config=config, system_prompt=system_prompt)
    state = runner.run(args.question)

    last_content = next(
        (m["content"] for m in reversed(state.messages) if m.get("role") == "assistant" and m.get("content")),
        None,
    )
    if last_content:
        print(last_content)
    else:
        print(f"(Agent finished with reason: {state.finish_reason})")


if __name__ == "__main__":
    main()
