from __future__ import annotations

import os

import pytest

from kgent.config.loader import SandboxConfig
from kgent.tools.static.python_exec import run_python_sandbox


def _cfg(**kwargs) -> SandboxConfig:
    defaults = dict(timeout_seconds=10.0, max_output_bytes=65536, allow_network=False)
    defaults.update(kwargs)
    return SandboxConfig(**defaults)


# --- basic execution ---

def test_basic_output():
    result = run_python_sandbox('print("hello")', _cfg())
    assert "hello" in result


def test_multiline_code():
    code = "x = 1 + 1\nprint(x)"
    result = run_python_sandbox(code, _cfg())
    assert "2" in result


def test_stderr_captured():
    code = "import sys; sys.stderr.write('err_output\\n')"
    result = run_python_sandbox(code, _cfg())
    assert "err_output" in result


def test_no_output_returns_placeholder():
    result = run_python_sandbox("x = 1", _cfg())
    assert result == "[no output]"


# --- error handling ---

def test_exception_captured():
    result = run_python_sandbox('raise ValueError("boom")', _cfg())
    assert "ValueError" in result
    assert "boom" in result


def test_exception_does_not_raise():
    # run_python_sandbox must never raise — it always returns a string
    result = run_python_sandbox("raise RuntimeError('fail')", _cfg())
    assert isinstance(result, str)


def test_syntax_error_captured():
    result = run_python_sandbox("def (broken:", _cfg())
    assert isinstance(result, str)
    assert len(result) > 0


# --- timeout ---

def test_timeout():
    result = run_python_sandbox("while True: pass", _cfg(timeout_seconds=0.5))
    assert "timed out" in result.lower() or "timeout" in result.lower()


def test_timeout_does_not_raise():
    result = run_python_sandbox("import time; time.sleep(999)", _cfg(timeout_seconds=0.5))
    assert isinstance(result, str)


# --- output truncation ---

def test_output_truncation():
    code = "print('x' * 200000)"
    result = run_python_sandbox(code, _cfg(max_output_bytes=1024))
    assert "OUTPUT TRUNCATED" in result
    assert len(result) <= 1024 + len("\n[OUTPUT TRUNCATED]") + 10  # small slack


def test_output_not_truncated_when_within_limit():
    result = run_python_sandbox('print("short")', _cfg(max_output_bytes=65536))
    assert "OUTPUT TRUNCATED" not in result


# --- env isolation ---

def test_no_env_leak():
    """No secret-like env var names must appear in the subprocess environment."""
    secret_patterns = ["KEY", "TOKEN", "SECRET", "PASSWORD", "AWS_", "OPENAI_", "ANTHROPIC_"]
    code = "import os, json; print(json.dumps(list(os.environ.keys())))"
    result = run_python_sandbox(code, _cfg())
    for pattern in secret_patterns:
        # Check if any env key containing this pattern leaked
        leaked = [k for k in result.split('"') if pattern in k.upper()]
        assert leaked == [], f"Secret-like env key leaked: {leaked}"


def test_home_is_tmpdir():
    """HOME should be set to a temp dir, not the real home directory."""
    real_home = os.environ.get("HOME", "")
    code = "import os; print(os.environ.get('HOME', 'NONE'))"
    result = run_python_sandbox(code, _cfg())
    if real_home:
        assert real_home not in result


# --- generated tool _INPUT injection ---

def test_input_injection():
    """Generated tool code must be able to read _INPUT."""
    from kgent.agent.types import ToolSource
    from kgent.tools.generated import make_save_as_tool
    from kgent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    save_tool = make_save_as_tool(registry, _cfg())

    # Register a generated tool that echoes its input
    result = save_tool.callable({
        "name": "echo_input",
        "description": "echoes value from _INPUT",
        "code": 'print(_INPUT["value"])',
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    })
    assert "registered" in result.lower()

    # Retrieve and invoke the generated tool
    echo_tool = registry.get("echo_input")
    assert echo_tool is not None
    assert echo_tool.source == ToolSource.GENERATED

    output = echo_tool.callable({"value": "injected_value"})
    assert "injected_value" in output


def test_input_injection_numeric():
    """_INPUT values of non-string types must survive serialisation round-trip."""
    from kgent.tools.generated import make_save_as_tool
    from kgent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    save_tool = make_save_as_tool(registry, _cfg())
    save_tool.callable({
        "name": "double_num",
        "description": "doubles a number",
        "code": "print(_INPUT['n'] * 2)",
        "input_schema": {"type": "object", "properties": {"n": {"type": "number"}}},
    })

    tool = registry.get("double_num")
    assert tool is not None
    output = tool.callable({"n": 21})
    assert "42" in output


# --- save_as_tool registration ---

def test_save_as_tool_returns_confirmation():
    from kgent.tools.generated import make_save_as_tool
    from kgent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    save_tool = make_save_as_tool(registry, _cfg())
    result = save_tool.callable({
        "name": "noop_tool",
        "description": "does nothing",
        "code": 'print("ok")',
        "input_schema": {"type": "object", "properties": {}},
    })
    assert "noop_tool" in result
    assert "registered" in result.lower()


def test_save_as_tool_overwrites_existing():
    """Registering a tool with an existing name replaces the old one."""
    from kgent.tools.generated import make_save_as_tool
    from kgent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    save_tool = make_save_as_tool(registry, _cfg())

    save_tool.callable({
        "name": "dup", "description": "v1", "code": 'print("v1")',
        "input_schema": {"type": "object", "properties": {}},
    })
    save_tool.callable({
        "name": "dup", "description": "v2", "code": 'print("v2")',
        "input_schema": {"type": "object", "properties": {}},
    })

    output = registry.get("dup").callable({})
    assert "v2" in output
