from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel


def _expand_env(value: str) -> str:
    """Replace ${VAR} with the value of os.environ['VAR']."""
    return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)


def _expand_env_in_dict(d: dict) -> dict:
    return {k: _expand_env(v) if isinstance(v, str) else v for k, v in d.items()}


class ModelConfig(BaseModel):
    provider: str
    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.0


class AgentCoreConfig(BaseModel):
    max_iterations: int = 20
    tool_timeout_seconds: float = 30.0
    repeated_call_window: int = 3


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = []
    env: dict[str, str] = {}

    def expanded_env(self) -> dict[str, str]:
        return _expand_env_in_dict(self.env)


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = []


class ToolsConfig(BaseModel):
    static: list[str] = []


class SandboxConfig(BaseModel):
    timeout_seconds: float = 15.0
    max_output_bytes: int = 65536
    allow_network: bool = False


class AgentPocConfig(BaseModel):
    model: ModelConfig
    agent: AgentCoreConfig
    tools: ToolsConfig
    mcp: MCPConfig = MCPConfig()
    sandbox: SandboxConfig = SandboxConfig()


def load_config(path: str | Path) -> AgentPocConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AgentPocConfig.model_validate(raw)
