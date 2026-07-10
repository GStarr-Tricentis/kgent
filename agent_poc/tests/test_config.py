from __future__ import annotations

import pytest

from agent_poc.config.loader import load_config


CONFIG_PATH = "agent_poc/config/config.yaml"


def test_load_config_model_name():
    c = load_config(CONFIG_PATH)
    assert c.model.model_name == "qwen2.5:7b"


def test_load_config_agent_defaults():
    c = load_config(CONFIG_PATH)
    assert c.agent.max_iterations == 20
    assert c.agent.repeated_call_window == 3


def test_load_config_sandbox():
    c = load_config(CONFIG_PATH)
    assert c.sandbox.timeout_seconds == 15
    assert c.sandbox.max_output_bytes == 65536
    assert c.sandbox.allow_network is False


def test_load_config_mcp_neo4j():
    c = load_config(CONFIG_PATH)
    assert len(c.mcp.servers) == 1
    assert c.mcp.servers[0].name == "neo4j"
    assert "NEO4J_URI" in c.mcp.servers[0].env


def test_load_config_mcp_env_expansion():
    import os
    os.environ["NEO4J_PASSWORD"] = "test_secret"
    c = load_config(CONFIG_PATH)
    expanded = c.mcp.servers[0].expanded_env()
    assert expanded["NEO4J_PASSWORD"] == "test_secret"
    del os.environ["NEO4J_PASSWORD"]
