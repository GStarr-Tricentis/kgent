"""
Integration tests — require a running Ollama instance.

Run with:
    pytest kgent/tests/integration/ -v -m integration

The default model is read from config.yaml. Override with:
    AGENT_MODEL=llama3.1:8b pytest kgent/tests/integration/ -v -m integration
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from kgent.agent.runner import AgentRunner
from kgent.config.loader import load_config
from kgent.models.openai_compatible import OpenAICompatibleBackend
from kgent.tools.generated import make_save_as_tool
from kgent.tools.registry import ToolRegistry
from kgent.tools.static.filesystem import LIST_DIR_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL
from kgent.tools.static.python_exec import make_python_exec_tool
from kgent.tools.static.shell import RUN_COMMAND_TOOL


CONFIG_PATH = "kgent/config/config.yaml"


def _make_runner(tmp_path: Path | None = None) -> AgentRunner:
    config = load_config(CONFIG_PATH)
    model_override = os.environ.get("AGENT_MODEL")
    if model_override:
        config.model.model_name = model_override

    system_prompt = Path("kgent/prompts/system.txt")
    prompt_text = system_prompt.read_text() if system_prompt.exists() else ""

    backend = OpenAICompatibleBackend(config.model)
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)
    registry.register(WRITE_FILE_TOOL)
    registry.register(LIST_DIR_TOOL)
    registry.register(RUN_COMMAND_TOOL)
    registry.register(make_python_exec_tool(config.sandbox))
    registry.register(make_save_as_tool(registry, config.sandbox))

    return AgentRunner(
        backend=backend,
        registry=registry,
        config=config,
        system_prompt=prompt_text,
    )


def _final_answer(state) -> str:
    for msg in reversed(state.messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


# ---------------------------------------------------------------------------
# Demo 1: agent reads a file and answers a question about its contents
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_demo1_read_file_and_answer(tmp_path):
    target = tmp_path / "cities.txt"
    target.write_text("Paris\nTokyo\nNairobi\nSydney\n")

    runner = _make_runner()
    state = runner.run(f"Read the file {target} and tell me how many cities are listed.")

    answer = _final_answer(state)
    assert "4" in answer or "four" in answer.lower(), (
        f"Expected count of 4 cities in answer, got: {answer!r}"
    )
    assert state.finished is True


# ---------------------------------------------------------------------------
# Demo 2: agent uses python_exec to perform a computation
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_demo2_python_exec_computation():
    runner = _make_runner()
    state = runner.run(
        "Use python_exec to compute the sum of the first 100 natural numbers and report the result."
    )

    answer = _final_answer(state)
    assert "5050" in answer, f"Expected 5050 in answer, got: {answer!r}"
    assert state.finished is True


# ---------------------------------------------------------------------------
# Demo 3: agent handles a custom file format by writing its own parser
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_demo4_custom_format_parser(tmp_path):
    data_file = tmp_path / "records.dat"
    data_file.write_text(textwrap.dedent("""\
        ##name=Alice;age=30;city=NYC
        ##name=Bob;age=25;city=LA
        ##name=Carol;age=35;city=Chicago
    """))

    runner = _make_runner()
    state = runner.run(
        f"Parse {data_file} — each line starts with ## and fields are separated by ; "
        f"in key=value format. How many records are there and what are the names?"
    )

    answer = _final_answer(state)
    assert "3" in answer or "three" in answer.lower(), (
        f"Expected record count 3, got: {answer!r}"
    )
    for name in ("Alice", "Bob", "Carol"):
        assert name in answer, f"Expected name {name!r} in answer, got: {answer!r}"
    assert state.finished is True


# ---------------------------------------------------------------------------
# Demo 4: save_as_tool creates a reusable generated tool
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_demo_save_as_tool_registers_generated_tool():
    from kgent.agent.types import ToolSource

    config = load_config(CONFIG_PATH)
    model_override = os.environ.get("AGENT_MODEL")
    if model_override:
        config.model.model_name = model_override

    prompt_text = Path("kgent/prompts/system.txt")
    prompt_text = prompt_text.read_text() if prompt_text.exists() else ""

    backend = OpenAICompatibleBackend(config.model)
    registry = ToolRegistry()
    registry.register(make_python_exec_tool(config.sandbox))
    registry.register(make_save_as_tool(registry, config.sandbox))

    runner = AgentRunner(
        backend=backend, registry=registry, config=config, system_prompt=prompt_text
    )

    state = runner.run(
        "Use save_as_tool to create a tool named 'add_numbers' that adds two numbers "
        "from _INPUT['a'] and _INPUT['b'] and prints the result."
    )

    assert state.finished is True
    tool = registry.get("add_numbers")
    assert tool is not None, "Expected 'add_numbers' to be registered after save_as_tool call"
    assert tool.source == ToolSource.GENERATED

    output = tool.callable({"a": 7, "b": 8})
    assert "15" in output, f"Expected 15 in tool output, got: {output!r}"
