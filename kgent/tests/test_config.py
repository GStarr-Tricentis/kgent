from __future__ import annotations

import pytest

from kgent.config.loader import load_config


CONFIG_PATH = "kgent/config/config.yaml"


def test_load_config_model_name():
    c = load_config(CONFIG_PATH)
    assert c.model.model_name


def test_load_config_agent_defaults():
    c = load_config(CONFIG_PATH)
    assert c.agent.max_iterations == 20


def test_load_config_sandbox():
    c = load_config(CONFIG_PATH)
    assert c.sandbox.timeout_seconds == 15
    assert c.sandbox.max_output_bytes == 65536
    assert c.sandbox.allow_network is False


def test_load_config_mcp_neo4j():
    c = load_config(CONFIG_PATH)
    assert len(c.mcp.servers) >= 1
    neo4j = next(s for s in c.mcp.servers if s.name == "neo4j")
    assert "NEO4J_URI" in neo4j.env


def test_load_config_mcp_env_expansion(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "test_secret")
    c = load_config(CONFIG_PATH)
    expanded = c.mcp.servers[0].expanded_env()
    assert expanded["NEO4J_PASSWORD"] == "test_secret"
