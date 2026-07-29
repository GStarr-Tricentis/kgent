from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys

import openai
from openai import AsyncOpenAI

from kgent.agent.types import ModelResponse, RegisteredTool, ToolCall

logger = logging.getLogger(__name__)


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
        self._is_anthropic = "anthropic." in deployment.lower()
        self._client: AsyncOpenAI | None = None
        self._async_anthropic_client = None
        self._tais_client = None
        self._auth_lock = asyncio.Lock()

    @classmethod
    async def create(cls, deployment: str, temperature: float = 0.0) -> "TricentisBackend":
        instance = cls(deployment, temperature)
        await instance.setup()
        return instance

    async def setup(self) -> None:
        from tricentis_ai_client import TaisClient, TaisConfig

        config = TaisConfig()
        client = TaisClient(config)
        try:
            await client.authenticate(interactive=False)
        except Exception:
            logger.warning(
                "\n[TAIS] Authentication required. "
                "Follow the link below to sign in via SSO:\n"
            )
            # The TAIS client prints the device-flow URL to stdout; redirect it
            # to stderr so it doesn't pollute captured output.
            with contextlib.redirect_stdout(sys.stderr):
                await client.authenticate(interactive=True)
        self._tais_client = client
        self._tais_config = config

        if self._is_anthropic:
            self._async_anthropic_client = client.create_anthropic_client(
                model=self._deployment,
            )
        else:
            self._client = AsyncOpenAI(
                base_url=f"{config.gateway_url}/api/v1/hub-service/openai/deployments/{self._deployment}",
                api_key=self._fresh_token(),
                default_headers={
                    "x-product-name": config.product_name,
                    "x-tenant-name": config.tenant_name,
                },
            )

    def _fresh_token(self) -> str:
        return self._tais_client.token_provider.get_valid_token()

    async def complete(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
        response_format: dict | None = None,
    ) -> ModelResponse:
        if self._is_anthropic:
            return await self._complete_anthropic(messages, tools)
        return await self._complete_openai(messages, tools)

    async def _complete_anthropic(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
        response_format: dict | None = None,
    ) -> ModelResponse:
        import anthropic

        system_text = ""
        anthropic_messages: list[dict] = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_text = msg["content"]
            elif role in ("user", "assistant") and "tool_calls" not in msg:
                anthropic_messages.append({"role": role, "content": msg["content"]})
            elif role == "assistant" and msg.get("tool_calls"):
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    })
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            elif role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }],
                })

        anthropic_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

        response = await self._async_anthropic_client.messages.create(
            model=self._deployment,
            messages=anthropic_messages,
            tools=anthropic_tools if anthropic_tools else anthropic.NOT_GIVEN,
            system=system_text if system_text else anthropic.NOT_GIVEN,
            max_tokens=8192,
            temperature=self._temperature,
            timeout=60.0,
        )

        text_content = next(
            (b.text for b in response.content if b.type == "text"), None
        )
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input)
            for b in response.content if b.type == "tool_use"
        ]
        finish_reason = {"end_turn": "stop", "tool_use": "tool_calls"}.get(
            response.stop_reason, response.stop_reason
        )

        assistant_message = {
            "role": "assistant",
            "content": text_content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in tool_calls
            ] or None,
        }

        return ModelResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw=response,
            assistant_message=assistant_message,
        )

    def _make_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=f"{self._tais_config.gateway_url}/api/v1/hub-service/openai/deployments/{self._deployment}",
            api_key=self._fresh_token(),
            default_headers={
                "x-product-name": self._tais_config.product_name,
                "x-tenant-name": self._tais_config.tenant_name,
            },
        )

    async def _reauthenticate(self) -> None:
        async with self._auth_lock:
            await self._tais_client.authenticate(interactive=False)

    async def _complete_openai(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
        response_format: dict | None = None,
    ) -> ModelResponse:
        client = self._make_openai_client()

        tool_payload = _tools_payload(tools)
        tools_param = tool_payload if tool_payload else openai.NOT_GIVEN

        try:
            response = await client.chat.completions.create(
                model=self._deployment,
                messages=messages,
                tools=tools_param,
                temperature=self._temperature,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError):
            await self._reauthenticate()
            client = self._make_openai_client()
            response = await client.chat.completions.create(
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
