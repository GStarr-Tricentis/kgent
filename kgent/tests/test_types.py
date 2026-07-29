from __future__ import annotations

import pytest

from kgent.agent.types import (
    ModelResponse,
    RegisteredTool,
    ToolCall,
    ToolResult,
    ToolSource,
)


def test_tool_source_values():
    assert ToolSource.STATIC == "static"
    assert ToolSource.MCP == "mcp"
    assert ToolSource.GENERATED == "generated"


def test_registered_tool_creation():
    tool = RegisteredTool(
        name="my_tool",
        description="does something",
        input_schema={"type": "object", "properties": {}},
        callable=lambda args: "ok",
        source=ToolSource.STATIC,
    )
    assert tool.name == "my_tool"
    assert tool.source == ToolSource.STATIC
    assert tool.timeout_seconds == 30.0


def test_registered_tool_custom_timeout():
    tool = RegisteredTool(
        name="t",
        description="",
        input_schema={},
        callable=lambda _: "",
        source=ToolSource.GENERATED,
        timeout_seconds=5.0,
    )
    assert tool.timeout_seconds == 5.0


def test_tool_call_creation():
    tc = ToolCall(id="abc", name="read_file", arguments={"path": "/tmp/x"})
    assert tc.id == "abc"
    assert tc.arguments == {"path": "/tmp/x"}


def test_tool_result_defaults():
    r = ToolResult(tool_call_id="x", name="shell", output="hello")
    assert r.error is False


def test_tool_result_error_flag():
    r = ToolResult(tool_call_id="x", name="shell", output="ERR", error=True)
    assert r.error is True


def test_model_response_no_tool_calls():
    r = ModelResponse(
        content="hi",
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": "hi"},
        raw=None,
    )
    assert r.content == "hi"
    assert r.tool_calls == []


def test_model_response_with_tool_calls():
    tc = ToolCall(id="1", name="foo", arguments={})
    r = ModelResponse(
        content=None,
        tool_calls=[tc],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": None},
        raw=object(),
    )
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "foo"
