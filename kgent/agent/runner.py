from __future__ import annotations

import asyncio
import json
import logging

from kgent.agent.state import RunState
from kgent.agent.types import ModelBackend, ToolResult
from kgent.config.loader import AgentPocConfig
from kgent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(
        self,
        backend: ModelBackend,
        registry: ToolRegistry,
        config: AgentPocConfig,
        system_prompt: str = "",
    ) -> None:
        self._backend = backend
        self._registry = registry
        self._config = config
        self._system_prompt = system_prompt

    async def run(self, user_input: str) -> RunState:
        state = RunState()

        if self._system_prompt:
            state.messages.append({"role": "system", "content": self._system_prompt})
        state.messages.append({"role": "user", "content": user_input})

        while state.iteration < self._config.agent.max_iterations:
            logger.debug("Iteration %d", state.iteration)

            response = await self._backend.complete(state.messages, self._registry.list_tools())

            state.messages.append(response.assistant_message)

            if response.finish_reason == "stop" and not response.tool_calls:
                state.finished = True
                state.finish_reason = "stop"
                state.iteration += 1
                break

            if response.finish_reason == "length":
                state.finish_reason = "length"
                state.iteration += 1
                break

            current_batch = [
                (tc.name, json.dumps(tc.arguments, sort_keys=True))
                for tc in response.tool_calls
            ]

            if current_batch == state.last_batch:
                results: list[ToolResult] = [
                    ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        output=(
                            f"Repeated identical tool call detected for '{tc.name}'. "
                            "Try a different approach."
                        ),
                        error=True,
                    )
                    for tc in response.tool_calls
                ]
            else:
                state.last_batch = current_batch
                results = list(
                    await asyncio.gather(
                        *[self._registry.execute(tc) for tc in response.tool_calls]
                    )
                )

            for result in results:
                state.execution_history.append(result)
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.output,
                })

            state.iteration += 1

        if not state.finished and not state.finish_reason:
            state.finish_reason = "max_iterations"

        return state
