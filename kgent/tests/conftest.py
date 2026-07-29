from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from kgent.agent.types import ModelResponse, ToolCall


class MockBackend:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def complete(self, messages: list[dict], tools: list, response_format: dict | None = None) -> ModelResponse:
        self.call_count += 1
        if not self._responses:
            raise RuntimeError("MockBackend exhausted — add more responses")
        return self._responses.pop(0)


def _make_raw(role: str = "assistant", content: str | None = None, tool_calls=None) -> MagicMock:
    raw = MagicMock()
    raw.choices[0].message.model_dump.return_value = {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
    }
    return raw


def make_stop_response(content: str = "Done.") -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content, "tool_calls": None},
        raw=_make_raw(content=content),
    )


def make_tool_call_response(
    tool_name: str,
    arguments: dict,
    call_id: str = "tc1",
) -> ModelResponse:
    tc = ToolCall(id=call_id, name=tool_name, arguments=arguments)
    return ModelResponse(
        content=None,
        tool_calls=[tc],
        finish_reason="tool_calls",
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "function": {"name": tool_name}}],
        },
        raw=_make_raw(
            tool_calls=[{"id": call_id, "function": {"name": tool_name}}]
        ),
    )
