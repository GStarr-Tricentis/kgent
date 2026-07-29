from __future__ import annotations

import shlex
import subprocess

from kgent.agent.types import RegisteredTool, ToolSource

MAX_OUTPUT = 8192
DEFAULT_TIMEOUT = 30


def _run_command(args: dict) -> str:
    command = args.get("command", "").strip()
    if not command:
        return "ERROR: empty command"

    timeout = args.get("_timeout_override", DEFAULT_TIMEOUT)
    cwd = args.get("cwd", None)

    try:
        cmd = shlex.split(command)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            shell=False,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            output = f"[exit code {result.returncode}]\n" + output
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + "\n[TRUNCATED]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {timeout}s."
    except Exception as exc:
        return f"ERROR: {exc}"


RUN_COMMAND_TOOL = RegisteredTool(
    name="shell",
    description=(
        "Run a shell command and return its stdout and stderr. "
        "Pipes and redirects are not supported (shell=False). "
        "Use the 'cwd' argument to set working directory."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Command to run. No pipes or shell redirects.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory.",
            },
        },
        "required": ["command"],
    },
    callable=_run_command,
    source=ToolSource.STATIC,
)
