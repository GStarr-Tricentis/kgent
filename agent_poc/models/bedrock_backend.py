from __future__ import annotations

import asyncio
import json

from agent_poc.agent.types import ModelResponse, RegisteredTool, ToolCall


class BedrockBackend:
    def __init__(self, model_id: str, region: str, temperature: float = 0.0) -> None:
        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id
        self._temperature = temperature

    async def complete(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
        response_format: dict | None = None,
    ) -> ModelResponse:
        # Convert messages: separate system messages and translate formats
        system_blocks: list[dict] = []
        bedrock_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_blocks.append({"text": content})

            elif role == "user":
                if isinstance(content, str):
                    bedrock_messages.append(
                        {"role": "user", "content": [{"text": content}]}
                    )
                else:
                    # already a list of content blocks
                    bedrock_messages.append({"role": "user", "content": content})

            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    tc_blocks = [
                        {
                            "toolUse": {
                                "toolUseId": tc["id"],
                                "name": tc["function"]["name"],
                                "input": json.loads(tc["function"]["arguments"]),
                            }
                        }
                        for tc in tool_calls
                    ]
                    text_blocks = [{"text": content}] if content else []
                    bedrock_messages.append(
                        {"role": "assistant", "content": tc_blocks + text_blocks}
                    )
                else:
                    bedrock_messages.append(
                        {"role": "assistant", "content": [{"text": content or ""}]}
                    )

            elif role == "tool":
                bedrock_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": msg["tool_call_id"],
                                    "content": [{"text": msg["content"]}],
                                }
                            }
                        ],
                    }
                )

        # Build tool config
        tool_config: dict | None = None
        if tools:
            tool_config = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": {"json": t.input_schema},
                        }
                    }
                    for t in tools
                ]
            }

        # Build kwargs for converse call
        kwargs: dict = {
            "modelId": self._model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {"temperature": self._temperature},
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if tool_config:
            kwargs["toolConfig"] = tool_config

        response = await asyncio.to_thread(self._client.converse, **kwargs)

        # Parse response
        stop_reason = response.get("stopReason", "end_turn")
        finish_reason = "tool_calls" if stop_reason == "tool_use" else "stop"

        output_content = response.get("output", {}).get("message", {}).get("content", [])

        text_content: str | None = None
        tool_calls: list[ToolCall] = []

        for block in output_content:
            if "text" in block:
                text_content = block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        id=tu["toolUseId"],
                        name=tu["name"],
                        arguments=tu["input"],
                    )
                )

        # Reconstruct assistant_message in OpenAI format
        assistant_message: dict = {
            "role": "assistant",
            "content": text_content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ]
            or None,
        }

        return ModelResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            assistant_message=assistant_message,
            raw=response,
        )
