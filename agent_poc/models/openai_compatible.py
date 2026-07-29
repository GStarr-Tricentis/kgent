from __future__ import annotations

import json

import openai
from openai import AsyncOpenAI

from agent_poc.agent.types import ModelResponse, RegisteredTool, ToolCall
from agent_poc.config.loader import ModelConfig


def _tools_payload(tools: list[RegisteredTool]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _parse_arguments(raw: str) -> dict:
    args = json.loads(raw)
    if isinstance(args, str):
        args = json.loads(args)
    return args


class OpenAICompatibleBackend:
    def __init__(self, config: ModelConfig) -> None:
        self._client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
        self._model = config.model_name
        self._temperature = config.temperature

    async def complete(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
        response_format: dict | None = None,
    ) -> ModelResponse:
        tool_payload = _tools_payload(tools)
        tools_param = tool_payload if tool_payload else openai.NOT_GIVEN

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools_param,
            temperature=self._temperature,
            response_format=response_format or openai.NOT_GIVEN,
        )

        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or "stop"

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = _parse_arguments(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return ModelResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            assistant_message=response.choices[0].message.model_dump(exclude_none=False),
            raw=response,
        )
