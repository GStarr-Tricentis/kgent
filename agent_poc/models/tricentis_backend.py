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
        self._is_anthropic = "anthropic." in deployment.lower()
        self._client: OpenAI | None = None
        self._anthropic_client = None
        asyncio.run(self._setup())

    async def _setup(self) -> None:
        import sys
        from tricentis_ai_client import TaisClient, TaisConfig
        from tricentis_ai_client.exceptions import InteractiveAuthRequiredError

        config = TaisConfig()
        client = TaisClient(config)
        try:
            await client.authenticate(interactive=False)
        except (InteractiveAuthRequiredError, Exception):
            # No cached token — run the device flow with output on stderr so it
            # is always visible (in the terminal where `streamlit run` was launched,
            # as well as in plain CLI usage).
            print(
                "\n[TAIS] Authentication required. "
                "Follow the link below to sign in via SSO:\n",
                file=sys.stderr,
                flush=True,
            )
            # Temporarily redirect stdout → stderr so the device flow's URL
            # and user-code prints are visible regardless of how the process
            # was launched (Streamlit swallows stdout).
            _real_stdout = sys.stdout
            sys.stdout = sys.stderr
            try:
                await client.authenticate(interactive=True)
            finally:
                sys.stdout = _real_stdout
        self._tais_client = client

        if self._is_anthropic:
            # create_anthropic_client wires PAYG session ID automatically
            self._async_anthropic_client = client.create_anthropic_client(
                model=self._deployment,
            )
        else:
            token = client.token_provider.get_valid_token()
            self._client = OpenAI(
                base_url=f"{config.gateway_url}/api/v1/hub-service/openai/deployments/{self._deployment}",
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
        response_format: dict | None = None,
    ) -> ModelResponse:
        if self._is_anthropic:
            return self._complete_anthropic(messages, tools)
        return self._complete_openai(messages, tools)

    def _complete_anthropic(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
        response_format: dict | None = None,
    ) -> ModelResponse:
        import asyncio
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

        async def _call():
            return await self._async_anthropic_client.messages.create(
                model=self._deployment,
                messages=anthropic_messages,
                tools=anthropic_tools if anthropic_tools else anthropic.NOT_GIVEN,
                system=system_text if system_text else anthropic.NOT_GIVEN,
                max_tokens=4096,
                temperature=self._temperature,
            )

        response = asyncio.run(_call())

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

    def _complete_openai(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
        response_format: dict | None = None,
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
