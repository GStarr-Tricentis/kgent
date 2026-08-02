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
        from kgent.agent.types import ModelResponse
        return ModelResponse(
            content=self._content,
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": self._content},
            raw=None,
        )


class CapturingBackend:
    """Backend that records the initial prompt of each call and returns canned responses."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.prompts: list[str] = []
        self._call_count = 0

    async def complete(self, messages, tools, response_format=None):
        from kgent.agent.types import ModelResponse
        self.prompts.append(next(m["content"] for m in messages if m["role"] == "user"))
        content = self._responses[self._call_count]
        self._call_count += 1
        return ModelResponse(
            content=content,
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": content},
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

    def test_warns_on_unmapped_association_edge_name(self):
        from graph_pipeline.context_store import (
            AssociationConfig,
            DatasetContext,
            DatasetNodeType,
            DatasetRelationshipType,
        )
        from graph_pipeline.schema_discovery import validate_proposed_context

        ctx = DatasetContext(
            dataset_id="test",
            node_types=[DatasetNodeType(name="TestCase", maps_to="TestCase", identity_key="uniqueId")],
            relationship_types=[
                DatasetRelationshipType(name="Coverage", maps_to="COVERS", **{"from": "TestCase", "to": "TestCase"})
            ],
            association_config=AssociationConfig(
                array_field="associations",
                edge_name_subfield="edgeName",
                partner_id_subfield="partnerId",
                direction_default="out",
            ),
        )
        sample = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "associations": [
                    {"edgeName": "Coverage", "partnerId": "tc-002"},
                    {"edgeName": "TestCase", "partnerId": "tc-003"},
                ],
            }
        ]

        warnings = validate_proposed_context(ctx, sample)
        assert len(warnings) == 1
        assert "TestCase" in warnings[0]
        assert all("Coverage" not in w for w in warnings)


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
                from kgent.agent.types import ModelResponse
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
                from kgent.agent.types import ModelResponse
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
                from kgent.agent.types import ModelResponse
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

    async def test_handled_fields_excludes_structural_fields(self):
        from graph_pipeline.context_store import SharedContext
        from graph_pipeline.schema_discovery import propose_dataset_context

        nodes_resp = json.dumps({
            "node_types": [{"name": "TestCase", "maps_to": "TestCase", "identity_key": "uniqueId"}],
            "id_field": "uniqueId",
            "type_field": "typeName",
            "hierarchy_config": {
                "field": "nodePath",
                "separator": "/",
                "phantom_label": "Folder",
                "edge_type": "CONTAINS",
            },
            "nested_collections": [
                {"field": "details.attrs", "child_label": "Attr", "edge_type": "HAS_ATTR", "id_field": "uniqueId"}
            ],
        })
        rels_resp = json.dumps({
            "relationship_types": [],
            "implicit_relationships": [
                {
                    "description": "x",
                    "pattern": "p",
                    "edge_name": "moduleUniqueId",
                    "maps_to": "USES_MODULE",
                    "from_type": "TestCase",
                    "to_type": "XModule",
                    "cross_dataset": False,
                    "target_dataset_id": None,
                }
            ],
            "association_config": {
                "array_field": "associations",
                "edge_name_subfield": "edgeName",
                "partner_id_subfield": "partnerId",
                "direction_subfield": None,
                "direction_default": "out",
            },
        })
        ambiguous_resp = json.dumps({"fields": []})

        backend = CapturingBackend([nodes_resp, rels_resp, ambiguous_resp])
        await propose_dataset_context(
            sample=SAMPLE,
            shared_context=SharedContext(),
            backend=backend,
        )

        ambiguous_prompt = backend.prompts[2]
        assert "nodePath" in ambiguous_prompt
        assert "details" in ambiguous_prompt
        assert "details.attrs" not in ambiguous_prompt
        assert "associations" in ambiguous_prompt
        assert "moduleUniqueId" in ambiguous_prompt

    async def test_hierarchy_field_note_in_relationships_prompt(self):
        from graph_pipeline.context_store import SharedContext
        from graph_pipeline.schema_discovery import propose_dataset_context

        minimal_rels = json.dumps({
            "relationship_types": [],
            "implicit_relationships": [],
            "association_config": None,
        })
        minimal_ambiguous = json.dumps({"fields": []})

        def _nodes_resp(with_hierarchy: bool) -> str:
            return json.dumps({
                "node_types": [{"name": "TestCase", "maps_to": "TestCase", "identity_key": "uniqueId"}],
                "id_field": "uniqueId",
                "type_field": "typeName",
                "hierarchy_config": (
                    {"field": "nodePath", "separator": "/", "phantom_label": "Folder", "edge_type": "CONTAINS"}
                    if with_hierarchy else None
                ),
                "nested_collections": [],
            })

        backend = CapturingBackend([_nodes_resp(True), minimal_rels, minimal_ambiguous])
        await propose_dataset_context(sample=SAMPLE, shared_context=SharedContext(), backend=backend)
        rels_prompt = backend.prompts[1]
        assert '"nodePath"' in rels_prompt
        assert "not record IDs" in rels_prompt

        backend2 = CapturingBackend([_nodes_resp(False), minimal_rels, minimal_ambiguous])
        await propose_dataset_context(sample=SAMPLE, shared_context=SharedContext(), backend=backend2)
        rels_prompt2 = backend2.prompts[1]
        assert "none identified" in rels_prompt2


# ---------------------------------------------------------------------------
# TestBuildFieldValueMatrix
# ---------------------------------------------------------------------------

class TestBuildFieldValueMatrix:
    def test_excludes_handled_fields(self):
        from graph_pipeline.schema_discovery import _build_field_value_matrix
        sample = [{"name": "Alice", "status": "active"}]
        result = _build_field_value_matrix(sample, handled_fields=["status"], id_field="id", type_field="type")
        assert "status" not in result
        assert "name" in result

    def test_excludes_id_and_type_fields(self):
        from graph_pipeline.schema_discovery import _build_field_value_matrix
        sample = [{"uniqueId": "u1", "typeName": "Foo", "label": "bar"}]
        result = _build_field_value_matrix(sample, handled_fields=[], id_field="uniqueId", type_field="typeName")
        assert "uniqueId" not in result
        assert "typeName" not in result
        assert "label" in result

    def test_excludes_nested_and_none(self):
        from graph_pipeline.schema_discovery import _build_field_value_matrix
        sample = [{"nested": {"a": 1}, "arr": [1, 2], "empty": None, "ok": "yes"}]
        result = _build_field_value_matrix(sample, handled_fields=[], id_field="id", type_field="type")
        assert "nested" not in result
        assert "arr" not in result
        assert "empty" not in result
        assert "ok" in result

    def test_excludes_id_suffix_fields(self):
        from graph_pipeline.schema_discovery import _build_field_value_matrix
        sample = [{"moduleId": "m1", "parentUniqueId": "p1", "title": "hello"}]
        result = _build_field_value_matrix(sample, handled_fields=[], id_field="id", type_field="type")
        assert "moduleId" not in result
        assert "parentUniqueId" not in result
        assert "title" in result

    def test_caps_at_n_values(self):
        from graph_pipeline.schema_discovery import _build_field_value_matrix
        sample = [{"tag": str(i)} for i in range(30)]
        result = _build_field_value_matrix(sample, handled_fields=[], id_field="id", type_field="type", n_values=25)
        assert len(result["tag"]) == 25

    def test_deduplicates_values(self):
        from graph_pipeline.schema_discovery import _build_field_value_matrix
        sample = [{"status": "open"}, {"status": "open"}, {"status": "closed"}]
        result = _build_field_value_matrix(sample, handled_fields=[], id_field="id", type_field="type")
        assert result["status"] == ["closed", "open"] or set(result["status"]) == {"open", "closed"}
        assert len(result["status"]) == 2


class TestFilterAmbiguousByUidCoverage:
    def test_field_with_no_uid_matches_filtered(self):
        from graph_pipeline.schema_discovery import _filter_ambiguous_by_uid_coverage

        sample = [
            {"uniqueId": "tc-001", "label": "Setup - Install JRE"},
            {"uniqueId": "tc-002", "label": "Device Compatibility"},
            {"uniqueId": "tc-003", "label": "Regression Suite"},
        ]
        result = _filter_ambiguous_by_uid_coverage(["label"], sample, id_field="uniqueId")
        assert result == []

    def test_field_with_high_uid_match_rate_kept(self):
        from graph_pipeline.schema_discovery import _filter_ambiguous_by_uid_coverage

        sample = [
            {"uniqueId": "tc-001", "relatedId": "tc-002"},
            {"uniqueId": "tc-002", "relatedId": "tc-003"},
            {"uniqueId": "tc-003", "relatedId": "tc-001"},
        ]
        result = _filter_ambiguous_by_uid_coverage(["relatedId"], sample, id_field="uniqueId")
        assert result == ["relatedId"]

    def test_empty_proposed_fields(self):
        from graph_pipeline.schema_discovery import _filter_ambiguous_by_uid_coverage

        sample = [{"uniqueId": "tc-001", "name": "Test"}]
        result = _filter_ambiguous_by_uid_coverage([], sample, id_field="uniqueId")
        assert result == []

    def test_threshold_boundary(self):
        from graph_pipeline.schema_discovery import _filter_ambiguous_by_uid_coverage

        # 4 records; field has 4 values: 2 match uniqueIds (50%) → kept at default 0.50
        sample = [
            {"uniqueId": "tc-001"},
            {"uniqueId": "tc-002"},
            {"uniqueId": "tc-003"},
            {"uniqueId": "tc-004"},
        ]
        for r, val in zip(sample, ["tc-001", "tc-002", "label-A", "label-B"]):
            r["ref"] = val

        result = _filter_ambiguous_by_uid_coverage(["ref"], sample, id_field="uniqueId")
        assert result == ["ref"]

        # Drop one match so only 1 of 4 (25%) matches → dropped
        sample[1]["ref"] = "label-C"
        result = _filter_ambiguous_by_uid_coverage(["ref"], sample, id_field="uniqueId")
        assert result == []

    def test_sample_with_no_uid_field(self):
        from graph_pipeline.schema_discovery import _filter_ambiguous_by_uid_coverage

        sample = [{"name": "Alice"}, {"name": "Bob"}]
        result = _filter_ambiguous_by_uid_coverage(["name"], sample, id_field="uniqueId")
        assert result == ["name"]


@pytest.mark.llm
async def test_propose_dataset_context_returns_valid_result():
    """Call a real model and assert the result is a structurally valid DatasetContext."""
    from graph_pipeline.context_store import DatasetContext, SharedContext
    from graph_pipeline.schema_discovery import propose_dataset_context
    from kgent.agent.backends.ollama import OllamaBackend

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
