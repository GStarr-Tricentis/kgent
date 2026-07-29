from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class ToolSource(str, Enum):
    STATIC = "static"
    MCP = "mcp"
    GENERATED = "generated"


class RegisteredTool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict
    callable: Any  # Callable[[dict], str]
    source: ToolSource
    timeout_seconds: float = 30.0


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    output: str
    error: bool = False


class ModelResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str  # "stop", "tool_calls", "length"
    assistant_message: dict
    raw: Any


class ModelBackend(Protocol):
    async def complete(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
        response_format: dict | None = None,
    ) -> ModelResponse: ...
