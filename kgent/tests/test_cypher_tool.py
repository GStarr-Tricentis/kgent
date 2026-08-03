from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kgent.agent.types import ModelResponse
from kgent.config.loader import (
    AgentCoreConfig,
    CypherToolConfig,
    KgentConfig,
    ModelConfig,
    ToolsConfig,
)
from kgent.tools.cypher_tool import (
    _extract_labels,
    _format_results,
    _known_labels,
    _make_undirected,
    _strip_fences,
    _trim_neo4j_error,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_STR = "Node labels:\n  `TestCase`\n  `TestSuite`\n"


def _make_config(**cypher_kwargs) -> KgentConfig:
    return KgentConfig(
        model=ModelConfig(
            provider="local",
            base_url="http://localhost:11434/v1",
            api_key="",
            model_name="test-model",
        ),
        agent=AgentCoreConfig(),
        tools=ToolsConfig(),
        cypher_tool=CypherToolConfig(**cypher_kwargs),
    )


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={},
        raw=None,
    )


@pytest.fixture
async def cypher_env(monkeypatch):
    """Patches all external deps and yields (callable, backend, driver, session)."""
    monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")

    backend = AsyncMock()
    driver = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value = []
    driver.session.return_value = session

    with (
        patch("neo4j.GraphDatabase.driver", return_value=driver),
        patch("kgent.models.factory.make_backend", new=AsyncMock(return_value=backend)),
        patch("kgent.tools.cypher_tool._get_cached_schema", new=AsyncMock(return_value=SCHEMA_STR)),
    ):
        from kgent.tools.cypher_tool import make_cypher_tool
        tool = await make_cypher_tool(_make_config())
        yield tool.callable, backend, driver, session


# ──────────────────────────────────────────────────────────────────────────────
# _strip_fences
# ──────────────────────────────────────────────────────────────────────────────

def test_strip_fences_no_fence():
    assert _strip_fences("MATCH (n) RETURN n") == "MATCH (n) RETURN n"


def test_strip_fences_cypher_block():
    assert _strip_fences("```cypher\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"


def test_strip_fences_plain_block():
    assert _strip_fences("```\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"


def test_strip_fences_strips_surrounding_whitespace():
    assert _strip_fences("  MATCH (n) RETURN n  ") == "MATCH (n) RETURN n"


# ──────────────────────────────────────────────────────────────────────────────
# _extract_labels
# ──────────────────────────────────────────────────────────────────────────────

def test_extract_labels_single():
    assert _extract_labels("MATCH (n:TestCase) RETURN n") == {"TestCase"}


def test_extract_labels_multiple():
    assert _extract_labels("MATCH (a:TestCase)--(b:TestSuite) RETURN a") == {"TestCase", "TestSuite"}


def test_extract_labels_ignores_rel_type():
    assert _extract_labels("MATCH (a:TestCase)-[:REL]->(b:TestSuite) RETURN a") == {"TestCase", "TestSuite"}


def test_extract_labels_empty():
    assert _extract_labels("MATCH (n) RETURN count(n)") == set()


# ──────────────────────────────────────────────────────────────────────────────
# _known_labels
# ──────────────────────────────────────────────────────────────────────────────

def test_known_labels_parses_backtick_labels():
    assert _known_labels("Node labels:\n  `TestCase`\n  `TestSuite`\n") == {"TestCase", "TestSuite"}


def test_known_labels_stops_at_next_section():
    schema = "Node labels:\n  `TestCase`\nRelationship patterns:\n  `Ghost`\n"
    assert _known_labels(schema) == {"TestCase"}


def test_known_labels_empty_schema():
    assert _known_labels("") == set()


# ──────────────────────────────────────────────────────────────────────────────
# _make_undirected
# ──────────────────────────────────────────────────────────────────────────────

def test_make_undirected_outgoing():
    assert _make_undirected("(a)-[r]->(b)") == "(a)-[r]-(b)"


def test_make_undirected_incoming():
    assert _make_undirected("(a)<-[r]-(b)") == "(a)-[r]-(b)"


def test_make_undirected_already_undirected():
    assert _make_undirected("(a)-[r]-(b)") == "(a)-[r]-(b)"


# ──────────────────────────────────────────────────────────────────────────────
# _format_results
# ──────────────────────────────────────────────────────────────────────────────

def test_format_results_empty():
    assert _format_results([]) == "No results found."


def test_format_results_single():
    out = _format_results([{"count": 42}])
    assert "Row 1: count=42" in out
    assert "(1 result)" in out


def test_format_results_plural():
    out = _format_results([{"n": "a"}, {"n": "b"}])
    assert "Row 2:" in out
    assert "(2 results)" in out


# ──────────────────────────────────────────────────────────────────────────────
# _trim_neo4j_error
# ──────────────────────────────────────────────────────────────────────────────

def test_trim_neo4j_error_strips_prefix():
    exc = Exception("Neo.ClientError.Statement.SyntaxError: bad syntax")
    assert _trim_neo4j_error(exc) == "bad syntax"


def test_trim_neo4j_error_truncates_long_message():
    exc = Exception("x" * 400)
    assert len(_trim_neo4j_error(exc)) == 300


def test_trim_neo4j_error_skips_blank_lines():
    exc = Exception("\n\nActual error here")
    assert _trim_neo4j_error(exc) == "Actual error here"


# ──────────────────────────────────────────────────────────────────────────────
# _query_graph — happy path
# ──────────────────────────────────────────────────────────────────────────────

async def test_successful_query_returns_formatted_results(cypher_env):
    fn, backend, _, session = cypher_env
    backend.complete.return_value = _response("MATCH (n:TestCase) RETURN count(n) AS total")
    session.run.return_value = [{"total": 7}]

    result = await fn({"question": "How many test cases?"})

    assert "7" in result
    assert "Row 1" in result


async def test_no_results_returns_no_results_found(cypher_env):
    fn, backend, _, session = cypher_env
    backend.complete.return_value = _response("MATCH (n:TestCase) RETURN n")
    session.run.return_value = []

    result = await fn({"question": "List test cases"})

    assert result == "No results found."


# ──────────────────────────────────────────────────────────────────────────────
# _query_graph — empty model response
# ──────────────────────────────────────────────────────────────────────────────

async def test_empty_response_returns_error(cypher_env):
    fn, backend, *_ = cypher_env
    backend.complete.return_value = _response("")

    result = await fn({"question": "anything"})

    assert "empty response" in result.lower()


# ──────────────────────────────────────────────────────────────────────────────
# _query_graph — query parameter guard
# ──────────────────────────────────────────────────────────────────────────────

async def test_query_parameter_returns_error_without_executing(cypher_env):
    fn, backend, driver, _ = cypher_env
    backend.complete.return_value = _response("MATCH (n:TestCase {id: $id}) RETURN n")

    result = await fn({"question": "find by id"})

    assert "query parameters" in result.lower()
    driver.session.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# _query_graph — label correction retry
# ──────────────────────────────────────────────────────────────────────────────

async def test_unknown_label_triggers_correction_retry(cypher_env):
    fn, backend, _, session = cypher_env
    backend.complete.side_effect = [
        _response("MATCH (n:UnknownLabel) RETURN n"),
        _response("MATCH (n:TestCase) RETURN count(n) AS total"),
    ]
    session.run.return_value = [{"total": 3}]

    result = await fn({"question": "count"})

    assert backend.complete.call_count == 2
    assert "3" in result


async def test_unknown_label_retry_prompt_contains_correction(cypher_env):
    fn, backend, _, session = cypher_env
    backend.complete.side_effect = [
        _response("MATCH (n:UnknownLabel) RETURN n"),
        _response("MATCH (n:TestCase) RETURN count(n) AS total"),
    ]
    session.run.return_value = [{"total": 1}]

    await fn({"question": "count"})

    _, retry_call = backend.complete.call_args_list
    messages = retry_call[0][0]
    assert any("CORRECTION" in str(m) for m in messages)


async def test_unknown_label_retry_empty_response_returns_error(cypher_env):
    fn, backend, *_ = cypher_env
    backend.complete.side_effect = [
        _response("MATCH (n:UnknownLabel) RETURN n"),
        _response(""),
    ]

    result = await fn({"question": "count"})

    assert "empty response" in result.lower()


# ──────────────────────────────────────────────────────────────────────────────
# _query_graph — Cypher execution error retry
# ──────────────────────────────────────────────────────────────────────────────

async def test_cypher_error_triggers_rewrite_retry(cypher_env):
    fn, backend, _, session = cypher_env
    backend.complete.side_effect = [
        _response("MATCH (n:TestCase) RETURN n"),
        _response("MATCH (n:TestCase) RETURN count(n) AS total"),
    ]
    session.run.side_effect = [
        Exception("SyntaxError: bad query"),
        [{"total": 5}],
    ]

    result = await fn({"question": "count"})

    assert backend.complete.call_count == 2
    assert "5" in result


async def test_cypher_error_retry_prompt_contains_error(cypher_env):
    fn, backend, _, session = cypher_env
    backend.complete.side_effect = [
        _response("MATCH (n:TestCase) RETURN n"),
        _response("MATCH (n:TestCase) RETURN count(n) AS total"),
    ]
    session.run.side_effect = [
        Exception("SyntaxError: bad query"),
        [{"total": 5}],
    ]

    await fn({"question": "count"})

    _, retry_call = backend.complete.call_args_list
    messages = retry_call[0][0]
    assert any("error" in str(m).lower() for m in messages)


async def test_cypher_error_retry_empty_response_returns_error(cypher_env):
    fn, backend, _, session = cypher_env
    backend.complete.side_effect = [
        _response("MATCH (n:TestCase) RETURN n"),
        _response(""),
    ]
    session.run.side_effect = Exception("SyntaxError: bad query")

    result = await fn({"question": "count"})

    assert "empty response" in result.lower()
