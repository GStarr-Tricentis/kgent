from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Open-weight LLM agent")
    parser.add_argument("--config", default="agent_poc/config/config.yaml")
    parser.add_argument("--model", default=None, help="Override model_name from config")
    parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--provider", default="local", choices=["local", "tricentis"],
                        help="Model provider (default: local)")
    parser.add_argument("--cypher-tool", action="store_true",
                        help="Use query_graph tool instead of raw Neo4j MCP tools")
    parser.add_argument("--raw-data", default=None, metavar="PATH",
                        help="Path to a JSONL dump of the source data. Tells the agent the file "
                             "exists and that it can build tools (via save_as_tool) to search it.")
    args = parser.parse_args()
    from agent_poc.config.loader import load_dotenv
    load_dotenv()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    from agent_poc.config.loader import load_config
    from agent_poc.agent.instrumentation import build_registry
    from agent_poc.agent.runner import AgentRunner
    from agent_poc.models.factory import make_backend

    config = load_config(args.config)
    if args.model:
        config.model.model_name = args.model
    system_prompt_path = Path("agent_poc/prompts/system.txt")
    system_prompt = system_prompt_path.read_text() if system_prompt_path.exists() else ""

    skip_servers = frozenset({"neo4j"}) if args.cypher_tool else frozenset()
    registry = await build_registry(config, skip_servers=skip_servers)

    async with registry:
        if args.cypher_tool:
            from agent_poc.tools.cypher_tool import make_cypher_tool
            registry.register(await make_cypher_tool(config))
            graph_prompt = Path("agent_poc/agent/prompts/text_to_cypher_tool.txt").read_text()
            system_prompt = system_prompt + ("\n\n" if system_prompt else "") + graph_prompt

        if args.raw_data:
            raw_data_prompt = (
                f"Raw source data is available as a JSONL file at: {args.raw_data}\n"
                "Each line is a JSON object representing an entity from the knowledge graph source data. "
                "You can read or search this file using your available tools (e.g. run_command, read_file, python_exec).\n\n"
                "You also have access to save_as_tool, which lets you define and register new Python tools at runtime. "
                "If you find a reusable operation useful—such as searching the data by keyword—consider building a tool "
                "for it rather than repeating ad-hoc steps."
            )
            system_prompt = system_prompt + ("\n\n" if system_prompt else "") + raw_data_prompt

        backend = await make_backend(config, provider=args.provider, model_override=args.model)
        runner = AgentRunner(backend=backend, registry=registry, config=config, system_prompt=system_prompt)

        def _reply(state) -> str:
            for msg in reversed(state.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    return msg["content"]
            return "[Agent stopped without a text response]"

        if args.prompt:
            print(_reply(await runner.run(args.prompt)))
            return
        print("Open-weight agent ready. Ctrl-C to exit.")
        while True:
            try:
                line = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if line:
                print(_reply(await runner.run(line)))


if __name__ == "__main__":
    asyncio.run(main())
