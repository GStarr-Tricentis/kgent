"""Tests for graph_pipeline/schema_discovery.py.

Unit tests run without a model. The integration test requires a live Ollama instance
and must be opted in with: pytest --llm tests/test_schema_discovery.py
"""
import json

import pytest


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------

class MockBackend:
    """Minimal ModelBackend implementation for unit tests."""

    def __init__(self, response_content: str):
        self._content = response_content

    async def complete(self, messages, tools, response_format=None):
        from agent_poc.agent.types import ModelResponse
        return ModelResponse(
            content=self._content,
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": self._content},
            raw=None,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE = [
    {
        "uniqueId": "tc-001",
        "typeName": "TestCase",
        "name": "Login Test",
        "nodePath": "Root/Suite A/Login Test",
        "moduleUniqueId": "xm-001",
        "associations": [
            {"edgeName": "Requirement", "partnerId": "req-001", "direction": "out"}
        ],
    },
    {
        "uniqueId": "tc-002",
        "typeName": "TestCase",
        "name": "Logout Test",
        "nodePath": "Root/Suite A/Logout Test",
        "moduleUniqueId": "xm-002",
        "associations": [],
    },
    {
        "uniqueId": "tc-003",
        "typeName": "TestCase",
        "name": "Register Test",
        "nodePath": "Root/Suite B/Register Test",
        "moduleUniqueId": "xm-001",
        "associations": [],
    },
    {
        "uniqueId": "xm-001",
        "typeName": "XModule",
        "name": "Login Module",
        "nodePath": "Root/Modules/Login Module",
        "associations": [],
    },
    {
        "uniqueId": "xm-002",
        "typeName": "XModule",
        "name": "Logout Module",
        "nodePath": "Root/Modules/Logout Module",
        "associations": [],
    },
]


# ---------------------------------------------------------------------------
# validate_proposed_context — unit tests (no model required)
# ---------------------------------------------------------------------------

class TestValidateProposedContext:
    def _make_ctx(self, node_type_names=None, rel_types=None):
        from graph_pipeline.context_store import (
            DatasetContext,
            DatasetNodeType,
            DatasetRelationshipType,
        )
        node_types = [
            DatasetNodeType(name=n, maps_to=n, identity_key="uniqueId")
            for n in (node_type_names or [])
        ]
        rel_types_objs = []
        for rt in (rel_types or []):
            rel_types_objs.append(
                DatasetRelationshipType(
                    name=rt["name"],
                    maps_to=rt["maps_to"],
                    **{"from": rt["from_type"], "to": rt["to_type"]},
                )
            )
        return DatasetContext(
            dataset_id="test",
            node_types=node_types,
            relationship_types=rel_types_objs,
        )

    def test_no_warnings_for_valid_context(self):
        from graph_pipeline.schema_discovery import validate_proposed_context
        ctx = self._make_ctx(node_type_names=["TestCase", "XModule"])
        warnings = validate_proposed_context(ctx, SAMPLE)
        # All referenced typeNames exist in sample — no warnings
        assert all("TestCase" not in w and "XModule" not in w for w in warnings)

    def test_warns_on_unknown_node_type(self):
        from graph_pipeline.schema_discovery import validate_proposed_context
        ctx = self._make_ctx(node_type_names=["TestCase", "Ghost"])
        warnings = validate_proposed_context(ctx, SAMPLE)
        assert any("Ghost" in w for w in warnings)

    def test_warns_on_relationship_unknown_from_type(self):
        from graph_pipeline.schema_discovery import validate_proposed_context
        ctx = self._make_ctx(
            node_type_names=["TestCase"],
            rel_types=[
                {"name": "COVERS", "maps_to": "COVERS", "from_type": "Unknown", "to_type": "TestCase"}
            ],
        )
        warnings = validate_proposed_context(ctx, SAMPLE)
        assert any("Unknown" in w for w in warnings)

    def test_warns_on_relationship_unknown_to_type(self):
        from graph_pipeline.schema_discovery import validate_proposed_context
        ctx = self._make_ctx(
            node_type_names=["TestCase"],
            rel_types=[
                {"name": "COVERS", "maps_to": "COVERS", "from_type": "TestCase", "to_type": "Ghost"}
            ],
        )
        warnings = validate_proposed_context(ctx, SAMPLE)
        assert any("Ghost" in w for w in warnings)

    def test_returns_list_of_strings(self):
        from graph_pipeline.schema_discovery import validate_proposed_context
        ctx = self._make_ctx(node_type_names=["TestCase"])
        result = validate_proposed_context(ctx, SAMPLE)
        assert isinstance(result, list)
        assert all(isinstance(w, str) for w in result)


# ---------------------------------------------------------------------------
# Integration test — requires live Ollama + --llm flag
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# propose_dataset_context — unit tests using MockBackend
# ---------------------------------------------------------------------------

class TestProposeDatasetContext:
    def _nodes_response(self):
        return json.dumps({
            "node_types": [
                {"name": "TestCase", "maps_to": "TestCase", "identity_key": "uniqueId"},
                {"name": "XModule", "maps_to": "XModule", "identity_key": "uniqueId"},
            ],
            "id_field": "uniqueId",
            "type_field": "typeName",
            "hierarchy_config": None,
            "nested_collections": [],
        })

    def _rels_response(self):
        return json.dumps({
            "relationship_types": [
                {"name": "Requirement", "maps_to": "COVERS", "from": "TestCase", "to": "Requirement"}
            ],
            "implicit_relationships": [],
            "association_config": None,
        })

    def _ambiguous_response(self):
        return json.dumps({"fields": []})

    async def test_returns_dataset_context(self):
        from graph_pipeline.context_store import DatasetContext, SharedContext
        from graph_pipeline.schema_discovery import propose_dataset_context

        # The mock needs to return different responses for 3 calls.
        responses = [self._nodes_response(), self._rels_response(), self._ambiguous_response()]
        call_count = [0]

        class MultiMockBackend:
            async def complete(self, messages, tools, response_format=None):
                from agent_poc.agent.types import ModelResponse
                content = responses[call_count[0]]
                call_count[0] += 1
                return ModelResponse(
                    content=content,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": content},
                    raw=None,
                )

        result = await propose_dataset_context(
            sample=SAMPLE,
            shared_context=SharedContext(),
            backend=MultiMockBackend(),
        )
        assert isinstance(result, DatasetContext)
        assert isinstance(result.node_types, list)
        assert isinstance(result.relationship_types, list)

    async def test_node_types_parsed(self):
        from graph_pipeline.context_store import SharedContext
        from graph_pipeline.schema_discovery import propose_dataset_context

        responses = [self._nodes_response(), self._rels_response(), self._ambiguous_response()]
        call_count = [0]

        class MultiMockBackend:
            async def complete(self, messages, tools, response_format=None):
                from agent_poc.agent.types import ModelResponse
                content = responses[call_count[0]]
                call_count[0] += 1
                return ModelResponse(
                    content=content,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": content},
                    raw=None,
                )

        result = await propose_dataset_context(
            sample=SAMPLE,
            shared_context=SharedContext(),
            backend=MultiMockBackend(),
        )
        names = {nt.name for nt in result.node_types}
        assert "TestCase" in names
        assert "XModule" in names

    async def test_relationship_types_parsed(self):
        from graph_pipeline.context_store import SharedContext
        from graph_pipeline.schema_discovery import propose_dataset_context

        responses = [self._nodes_response(), self._rels_response(), self._ambiguous_response()]
        call_count = [0]

        class MultiMockBackend:
            async def complete(self, messages, tools, response_format=None):
                from agent_poc.agent.types import ModelResponse
                content = responses[call_count[0]]
                call_count[0] += 1
                return ModelResponse(
                    content=content,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": content},
                    raw=None,
                )

        result = await propose_dataset_context(
            sample=SAMPLE,
            shared_context=SharedContext(),
            backend=MultiMockBackend(),
        )
        assert len(result.relationship_types) == 1
        assert result.relationship_types[0].maps_to == "COVERS"


@pytest.mark.llm
async def test_propose_dataset_context_returns_valid_result():
    """Call a real model and assert the result is a structurally valid DatasetContext."""
    from graph_pipeline.context_store import DatasetContext, SharedContext
    from graph_pipeline.schema_discovery import propose_dataset_context
    from agent_poc.agent.backends.ollama import OllamaBackend

    shared_ctx = SharedContext()
    result = propose_dataset_context(
        sample=SAMPLE,
        shared_context=shared_ctx,
        backend=OllamaBackend(model="qwen3:8b", base_url="http://localhost:11434/v1"),
    )
    assert isinstance(result, DatasetContext)
    assert isinstance(result.node_types, list)
    assert isinstance(result.relationship_types, list)
    # The model should at minimum recognise the two typeNames present
    proposed_names = {nt.name for nt in result.node_types}
    assert len(proposed_names) >= 1
