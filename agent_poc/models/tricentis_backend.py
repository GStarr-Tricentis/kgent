from __future__ import annotations

import asyncio
import json

import openai
from openai import OpenAI

from agent_poc.agent.types import ModelResponse, RegisteredTool, ToolCall


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


class TricentisBackend:
    def __init__(self, deployment: str, temperature: float = 0.0) -> None:
        self._deployment = deployment
        self._temperature = temperature
        self._client: OpenAI = asyncio.run(self._setup())

    async def _setup(self) -> OpenAI:
        from tricentis_ai_client import TaisClient, TaisConfig

        config = TaisConfig()
        client = TaisClient(config)
        await client.authenticate()
        token = client.token_provider.get_valid_token()
        self._tais_client = client
        return OpenAI(
            base_url=f"{config.gateway_url}/api/v1/hub-service/openai",
            api_key=token,
            default_headers={
                "x-product-name": config.product_name,
                "x-tenant-name": config.tenant_name,
            },
        )

    def _fresh_token(self) -> str:
        return self._tais_client.token_provider.get_valid_token()

    def complete(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
    ) -> ModelResponse:
        self._client.api_key = self._fresh_token()

        tool_payload = _tools_payload(tools)
        tools_param = tool_payload if tool_payload else openai.NOT_GIVEN

        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            tools=tools_param,
            temperature=self._temperature,
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
            raw=response,
            assistant_message=message.model_dump(exclude_none=False),
        )
