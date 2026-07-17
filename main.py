from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-weight LLM agent")
    parser.add_argument("--config", default="agent_poc/config/config.yaml")
    parser.add_argument("--model", default=None, help="Override model_name from config")
    parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--provider", default="local", choices=["local", "tricentis"],
                        help="Model provider (default: local)")
    parser.add_argument("--cypher-tool", action="store_true",
                        help="Use query_graph tool instead of raw Neo4j MCP tools")
    args = parser.parse_args()
    from agent_poc.config.loader import load_dotenv
    load_dotenv()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    from agent_poc.config.loader import load_config
    from agent_poc.tools.registry import ToolRegistry
    from agent_poc.tools.static.filesystem import LIST_DIR_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL
    from agent_poc.tools.static.shell import RUN_COMMAND_TOOL
    from agent_poc.tools.static.python_exec import make_python_exec_tool
    from agent_poc.tools.generated import make_save_as_tool
    from agent_poc.tools.mcp_adapter import MCP_AVAILABLE, MCPAdapter
    from agent_poc.agent.runner import AgentRunner

    config = load_config(args.config)
    if args.model:
        config.model.model_name = args.model
    system_prompt_path = Path("agent_poc/prompts/system.txt")
    system_prompt = system_prompt_path.read_text() if system_prompt_path.exists() else ""

    registry = ToolRegistry()
    _static = {"filesystem": [READ_FILE_TOOL, WRITE_FILE_TOOL, LIST_DIR_TOOL],
               "shell": [RUN_COMMAND_TOOL],
               "python_exec": [make_python_exec_tool(config.sandbox)]}
    for name in config.tools.static:
        for tool in _static.get(name, []):
            registry.register(tool)
    registry.register(make_save_as_tool(registry, config.sandbox))

    if MCP_AVAILABLE:
        for srv in config.mcp.servers:
            if args.cypher_tool and srv.name == "neo4j":
                continue
            try:
                adapter = MCPAdapter(srv.name, srv.command, srv.args, srv.expanded_env())
                adapter.connect()
                mcp_tools = adapter.list_tools()
                for t in mcp_tools:
                    registry.register(t)
                print(f"[mcp] {srv.name}: registered {len(mcp_tools)} tools")
            except Exception as e:
                print(f"[mcp] {srv.name} failed: {e}", file=sys.stderr)
    elif config.mcp.servers:
        print("[mcp] Warning: mcp package not installed", file=sys.stderr)

    if args.cypher_tool:
        from agent_poc.tools.cypher_tool import make_cypher_tool
        registry.register(make_cypher_tool(config))
        graph_prompt = Path("agent_poc/agent/prompts/text_to_cypher_tool.txt").read_text()
        system_prompt = system_prompt + ("\n\n" if system_prompt else "") + graph_prompt

    from agent_poc.models.factory import make_backend
    runner = AgentRunner(backend=make_backend(config, provider=args.provider, model_override=args.model),
                         registry=registry, config=config, system_prompt=system_prompt)

    def _reply(state) -> str:
        for msg in reversed(state.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return "[Agent stopped without a text response]"

    if args.prompt:
        print(_reply(runner.run(args.prompt)))
        return
    print("Open-weight agent ready. Ctrl-C to exit.")
    while True:
        try:
            line = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if line:
            print(_reply(runner.run(line)))


if __name__ == "__main__":
    main()
