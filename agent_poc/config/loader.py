from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


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


class GraphPipelineConfig(BaseModel):
    context_dir: str = "context/"
    default_sample_size: int = 50
    default_batch_size: int = 500
    default_model: str = "qwen3:8b"


class TricentisConfig(BaseModel):
    deployment: str = ""  # model deployment name; override with --model flag


class CypherToolConfig(BaseModel):
    provider: str = "local"
    model: str = ""  # falls back to model.model_name if empty
    timeout_seconds: float = 120.0


class AgentPocConfig(BaseModel):
    model: ModelConfig
    agent: AgentCoreConfig
    tools: ToolsConfig
    mcp: MCPConfig = MCPConfig()
    sandbox: SandboxConfig = SandboxConfig()
    graph_pipeline: GraphPipelineConfig = GraphPipelineConfig()
    tricentis: TricentisConfig = TricentisConfig()
    cypher_tool: CypherToolConfig = CypherToolConfig()


def _expand_env_in_raw(obj):
    """Recursively expand ${VAR} in all string values of a nested dict/list."""
    if isinstance(obj, dict):
        return {k: _expand_env_in_raw(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_in_raw(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj


def load_config(path: str | Path) -> AgentPocConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AgentPocConfig.model_validate(_expand_env_in_raw(raw))
