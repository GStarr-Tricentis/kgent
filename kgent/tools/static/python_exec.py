from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from kgent.agent.types import RegisteredTool, ToolSource
from kgent.config.loader import SandboxConfig

# Keys whose names contain any of these patterns are excluded from the
# subprocess env to prevent leaking credentials into untrusted code.
_SECRET_PATTERNS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AWS_", "OPENAI_", "ANTHROPIC_")


def _is_secret_key(name: str) -> bool:
    upper = name.upper()
    return any(pat in upper for pat in _SECRET_PATTERNS)


def run_python_sandbox(code: str, sandbox_config: SandboxConfig) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "script.py")
        with open(script_path, "w") as f:
            f.write(code)

        # Build env from an explicit allowlist — never os.environ.copy(),
        # which would leak API keys and other secrets into untrusted code.
        env: dict[str, str] = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": tmpdir,
            "TMPDIR": tmpdir,
        }
        path = os.environ.get("PATH", "")
        if path:
            env["PATH"] = path

        # Pull in non-secret env vars that Python tooling may need.
        for k, v in os.environ.items():
            if k not in env and not _is_secret_key(k):
                env[k] = v

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=sandbox_config.timeout_seconds,
                cwd=tmpdir,
                env=env,
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return f"[ERROR] Script timed out after {sandbox_config.timeout_seconds}s"
        except Exception as e:
            return f"[ERROR] Failed to run script: {e}"

        if len(output) > sandbox_config.max_output_bytes:
            output = output[:sandbox_config.max_output_bytes] + "\n[OUTPUT TRUNCATED]"
        return output if output else "[no output]"


def make_python_exec_tool(sandbox_config: SandboxConfig) -> RegisteredTool:
    def _exec(args: dict) -> str:
        return run_python_sandbox(args["code"], sandbox_config)

    return RegisteredTool(
        name="python_exec",
        description=(
            "Execute Python code in an isolated subprocess sandbox and return stdout/stderr. "
            "Print your results. Only stdlib is available. "
            "Generated tools receive their input via the _INPUT dict."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Print results to stdout.",
                },
            },
            "required": ["code"],
        },
        callable=_exec,
        source=ToolSource.STATIC,
    )


# Module-level constant kept for backwards-compat imports; use make_python_exec_tool in main.
PYTHON_EXEC_TOOL = make_python_exec_tool(SandboxConfig())
