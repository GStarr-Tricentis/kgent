from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_poc.agent.types import ToolResult


class RunState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[dict] = Field(default_factory=list)
    iteration: int = 0
    execution_history: list[ToolResult] = Field(default_factory=list)
    last_batch: list | None = None
    finished: bool = False
    finish_reason: str = ""


# Alias used by the phase 0/1 acceptance checklist
AgentState = RunState
