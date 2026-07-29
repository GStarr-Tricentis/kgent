from __future__ import annotations

import json

from kgent.agent.types import RegisteredTool, ToolSource
from kgent.config.loader import SandboxConfig
from kgent.tools.registry import ToolRegistry
from kgent.tools.static.python_exec import run_python_sandbox

_SAVE_AS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Unique snake_case name for the new tool.",
        },
        "description": {
            "type": "string",
            "description": "What the tool does.",
        },
        "code": {
            "type": "string",
            "description": (
                "Python code for the tool body. "
                "Access inputs via the _INPUT dict. Always print the result."
            ),
        },
        "input_schema": {
            "type": "object",
            "description": "JSON Schema describing the tool's input arguments.",
        },
    },
    "required": ["name", "description", "code", "input_schema"],
}


def make_save_as_tool(registry: ToolRegistry, sandbox_config: SandboxConfig) -> RegisteredTool:
    """Return the save_as_tool RegisteredTool, pre-bound to the registry and sandbox config."""

    def _save_as_tool(args: dict) -> str:
        name: str = args["name"]
        description: str = args["description"]
        user_code: str = args["code"]
        input_schema: dict = args["input_schema"]

        def _generated_callable(call_args: dict) -> str:
            # Serialize call_args as JSON, then embed as a JSON string literal so
            # the generated script can parse it with json.loads.
            serialized = json.dumps(json.dumps(call_args))
            preamble = (
                "import json as _json\n"
                f"_INPUT = _json.loads({serialized})\n"
            )
            return run_python_sandbox(preamble + user_code, sandbox_config)

        registry.register(RegisteredTool(
            name=name,
            description=description,
            input_schema=input_schema,
            callable=_generated_callable,
            source=ToolSource.GENERATED,
        ))
        return f"Tool '{name}' registered successfully."

    return RegisteredTool(
        name="save_as_tool",
        description=(
            "Save a Python snippet as a reusable tool. "
            "The tool's input is available inside the code via the _INPUT dict. "
            "Always print the result so it is captured as the tool's output."
        ),
        input_schema=_SAVE_AS_TOOL_SCHEMA,
        callable=_save_as_tool,
        source=ToolSource.STATIC,
    )
