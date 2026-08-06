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
        assert result == [("relatedId", "")]

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
        assert result == [("ref", "")]

        # Drop one match so only 1 of 4 (25%) matches → dropped
        sample[1]["ref"] = "label-C"
        result = _filter_ambiguous_by_uid_coverage(["ref"], sample, id_field="uniqueId")
        assert result == []

    def test_sample_with_no_uid_field(self):
        from graph_pipeline.schema_discovery import _filter_ambiguous_by_uid_coverage

        sample = [{"name": "Alice"}, {"name": "Bob"}]
        result = _filter_ambiguous_by_uid_coverage(["name"], sample, id_field="uniqueId")
        assert result == [("name", "")]


class TestFilterAmbiguousDelimited:
    def test_filter_keeps_delimited_field(self):
        from graph_pipeline.schema_discovery import _filter_ambiguous_by_uid_coverage

        sample = [
            {"id": "uid1", "linkedItems": "uid2,uid3"},
            {"id": "uid2", "linkedItems": "uid1,uid3"},
            {"id": "uid3", "linkedItems": "uid1,uid2"},
        ]
        result = _filter_ambiguous_by_uid_coverage(["linkedItems"], sample, id_field="id")
        assert result == [("linkedItems", ",")]

    def test_filter_drops_prose_field(self):
        from graph_pipeline.schema_discovery import _filter_ambiguous_by_uid_coverage

        sample = [
            {"id": "uid1", "notes": "this is a description"},
            {"id": "uid2", "notes": "another prose value here"},
        ]
        result = _filter_ambiguous_by_uid_coverage(["notes"], sample, id_field="id")
        assert result == []


class TestScanNestedCollectionCandidates:
    def test_top_level_array_with_id_field_detected(self):
        from graph_pipeline.schema_discovery import _scan_nested_collection_candidates
        sample = [
            {"uniqueId": "r1", "moduleAttributes": [{"uniqueId": "a1", "name": "x"}]},
        ]
        candidates = _scan_nested_collection_candidates(sample, id_field="uniqueId")
        fields = [c["field"] for c in candidates]
        assert "moduleAttributes" in fields

    def test_nested_dict_array_with_id_field_detected_as_dot_path(self):
        from graph_pipeline.schema_discovery import _scan_nested_collection_candidates
        sample = [
            {"uniqueId": "r1", "details": {"steps": [{"uniqueId": "s1", "action": "click"}]}},
        ]
        candidates = _scan_nested_collection_candidates(sample, id_field="uniqueId")
        fields = [c["field"] for c in candidates]
        assert "details.steps" in fields

    def test_array_of_primitives_not_detected(self):
        from graph_pipeline.schema_discovery import _scan_nested_collection_candidates
        sample = [{"uniqueId": "r1", "tags": ["a", "b", "c"]}]
        candidates = _scan_nested_collection_candidates(sample, id_field="uniqueId")
        assert candidates == []

    def test_array_of_dicts_without_id_field_not_detected(self):
        from graph_pipeline.schema_discovery import _scan_nested_collection_candidates
        sample = [{"uniqueId": "r1", "refs": [{"name": "x"}, {"name": "y"}]}]
        candidates = _scan_nested_collection_candidates(sample, id_field="uniqueId")
        assert candidates == []

    def test_records_with_field_counts_records_not_items(self):
        from graph_pipeline.schema_discovery import _scan_nested_collection_candidates
        sample = [
            {"uniqueId": "r1", "attrs": [{"uniqueId": "a1"}, {"uniqueId": "a2"}]},
            {"uniqueId": "r2", "attrs": [{"uniqueId": "a3"}]},
            {"uniqueId": "r3"},
        ]
        candidates = _scan_nested_collection_candidates(sample, id_field="uniqueId")
        match = next(c for c in candidates if c["field"] == "attrs")
        assert match["records_with_field"] == 2

    def test_empty_sample_returns_empty(self):
        from graph_pipeline.schema_discovery import _scan_nested_collection_candidates
        assert _scan_nested_collection_candidates([], id_field="uniqueId") == []


class TestValidateAssociationPartnerTypes:
    _assoc_config = {
        "array_field": "associations",
        "edge_name_subfield": "edgeName",
        "partner_id_subfield": "partnerUniqueId",
    }

    def _make_sample(self):
        return [
            {"uniqueId": "tc-1", "typeName": "TestCase",
             "associations": [
                 {"edgeName": "Module", "partnerUniqueId": "xm-1"},
                 {"edgeName": "Module", "partnerUniqueId": "am-1"},
             ]},
            {"uniqueId": "xm-1", "typeName": "XModule"},
            {"uniqueId": "am-1", "typeName": "ApiModule"},
        ]

    def test_heterogeneous_partners_clears_to_type(self):
        from graph_pipeline.schema_discovery import _validate_association_partner_types
        from graph_pipeline.context_store import DatasetRelationshipType
        rel_types = [
            DatasetRelationshipType(
                name="Module", maps_to="USES_MODULE",
                from_type="TestCase", to_type="XModule",
            )
        ]
        result = _validate_association_partner_types(
            rel_types, self._assoc_config, self._make_sample(),
            id_field="uniqueId", type_field="typeName",
        )
        assert result[0].to_type == ""

    def test_uniform_partners_preserves_to_type(self):
        from graph_pipeline.schema_discovery import _validate_association_partner_types
        from graph_pipeline.context_store import DatasetRelationshipType
        sample = [
            {"uniqueId": "tc-1", "typeName": "TestCase",
             "associations": [
                 {"edgeName": "Coverage", "partnerUniqueId": "req-1"},
                 {"edgeName": "Coverage", "partnerUniqueId": "req-2"},
             ]},
            {"uniqueId": "req-1", "typeName": "Requirement"},
            {"uniqueId": "req-2", "typeName": "Requirement"},
        ]
        rel_types = [
            DatasetRelationshipType(
                name="Coverage", maps_to="COVERS",
                from_type="Requirement", to_type="TestCase",
            )
        ]
        result = _validate_association_partner_types(
            rel_types, self._assoc_config, sample,
            id_field="uniqueId", type_field="typeName",
        )
        assert result[0].to_type == "TestCase"

    def test_empty_assoc_config_returns_unchanged(self):
        from graph_pipeline.schema_discovery import _validate_association_partner_types
        from graph_pipeline.context_store import DatasetRelationshipType
        rel_types = [
            DatasetRelationshipType(
                name="Module", maps_to="USES_MODULE",
                from_type="TestCase", to_type="XModule",
            )
        ]
        result = _validate_association_partner_types(
            rel_types, {}, self._make_sample(),
            id_field="uniqueId", type_field="typeName",
        )
        assert result[0].to_type == "XModule"

    def test_partner_not_in_sample_preserves_to_type(self):
        from graph_pipeline.schema_discovery import _validate_association_partner_types
        from graph_pipeline.context_store import DatasetRelationshipType
        sample = [
            {"uniqueId": "tc-1", "typeName": "TestCase",
             "associations": [{"edgeName": "Module", "partnerUniqueId": "xm-99"}]},
            # xm-99 not in sample — type unknown
        ]
        rel_types = [
            DatasetRelationshipType(
                name="Module", maps_to="USES_MODULE",
                from_type="TestCase", to_type="XModule",
            )
        ]
        result = _validate_association_partner_types(
            rel_types, self._assoc_config, sample,
            id_field="uniqueId", type_field="typeName",
        )
        assert result[0].to_type == "XModule"


class TestValidatePathFkFromTypes:
    def _make_pfk(self, container_path, from_type, maps_to="STEP_USES_MODULE"):
        from graph_pipeline.context_store import PathFKRelationship
        return PathFKRelationship(
            container_path=container_path,
            fk_field="moduleUniqueId",
            target_field="uniqueId",
            maps_to=maps_to,
            from_type=from_type,
            to_type="",
        )

    def test_correct_from_type_preserved(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "typeName": "RTSB",
             "details": {"testSteps": [{"uniqueId": "s1"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "RTSB")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == "RTSB"

    def test_wrong_from_type_cleared(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "typeName": "RTSB",
             "details": {"testSteps": [{"uniqueId": "s1"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "TestCase")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == ""

    def test_empty_from_type_unchanged(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "typeName": "RTSB",
             "details": {"testSteps": [{"uniqueId": "s1"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == ""

    def test_null_container_path_unchanged(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        pfk = self._make_pfk(None, "TestCase")
        result = _validate_path_fk_from_types(
            [pfk], [{"uniqueId": "r1", "typeName": "TestCase"}], "typeName"
        )
        assert result[0].from_type == "TestCase"

    def test_multiple_types_have_array_from_type_matches_one(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "typeName": "RTSB",
             "details": {"testSteps": [{"uniqueId": "s1"}]}},
            {"uniqueId": "r2", "typeName": "TestCase",
             "details": {"testSteps": [{"uniqueId": "s2"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "RTSB")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == "RTSB"

    def test_empty_array_not_counted(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "typeName": "TestCase",
             "details": {"testSteps": []}},
            {"uniqueId": "r2", "typeName": "RTSB",
             "details": {"testSteps": [{"uniqueId": "s1"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "TestCase")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == ""

    def test_no_type_field_in_sample_returns_unchanged(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "details": {"testSteps": [{"uniqueId": "s1"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "TestCase")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == "TestCase"

    def test_empty_path_fk_rels_returns_empty(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        assert _validate_path_fk_from_types([], [], "typeName") == []


class TestScanAssociationEdgeNames:
    def test_detects_edge_names_from_associations_array(self):
        from graph_pipeline.schema_discovery import _scan_association_edge_names
        sample = [
            {"uniqueId": "r1", "associations": [
                {"edgeName": "Coverage", "partnerUniqueId": "r2"},
                {"edgeName": "Module", "partnerUniqueId": "r3"},
            ]},
            {"uniqueId": "r2", "associations": [
                {"edgeName": "Coverage", "partnerUniqueId": "r1"},
            ]},
        ]
        result = _scan_association_edge_names(sample)
        assert result["array_field"] == "associations"
        assert result["edge_name_subfield"] == "edgeName"
        assert result["edge_names"] == ["Coverage", "Module"]

    def test_detects_partner_id_subfield(self):
        from graph_pipeline.schema_discovery import _scan_association_edge_names
        sample = [
            {"uniqueId": "r1", "associations": [
                {"edgeName": "Coverage", "partnerUniqueId": "r2"},
            ]},
        ]
        result = _scan_association_edge_names(sample)
        assert result["partner_id_subfield"] == "partnerUniqueId"

    def test_returns_empty_when_no_association_array(self):
        from graph_pipeline.schema_discovery import _scan_association_edge_names
        sample = [{"uniqueId": "r1", "name": "foo"}]
        assert _scan_association_edge_names(sample) == {}

    def test_ignores_non_dict_items_in_array(self):
        from graph_pipeline.schema_discovery import _scan_association_edge_names
        sample = [
            {"uniqueId": "r1", "associations": ["not-a-dict", 42, None]},
        ]
        assert _scan_association_edge_names(sample) == {}

    def test_deduplicates_edge_names_across_records(self):
        from graph_pipeline.schema_discovery import _scan_association_edge_names
        sample = [
            {"uniqueId": "r1", "associations": [{"edgeName": "Coverage", "partnerUniqueId": "r2"}]},
            {"uniqueId": "r2", "associations": [{"edgeName": "Coverage", "partnerUniqueId": "r1"}]},
            {"uniqueId": "r3", "associations": [{"edgeName": "Module", "partnerUniqueId": "r1"}]},
        ]
        result = _scan_association_edge_names(sample)
        assert result["edge_names"] == ["Coverage", "Module"]

    def test_partner_id_subfield_none_when_no_known_field(self):
        from graph_pipeline.schema_discovery import _scan_association_edge_names
        sample = [
            {"uniqueId": "r1", "associations": [
                {"edgeName": "Coverage", "weirdField": "r2"},
            ]},
        ]
        result = _scan_association_edge_names(sample)
        assert result["edge_names"] == ["Coverage"]
        assert result["partner_id_subfield"] is None

    def test_empty_sample_returns_empty(self):
        from graph_pipeline.schema_discovery import _scan_association_edge_names
        assert _scan_association_edge_names([]) == {}


class TestResolveAmbiguousFieldRules:
    def _make_backend(self, content: str):
        class _MockResponse:
            def __init__(self, c):
                self.content = c
                self.tool_calls = []
                self.finish_reason = "stop"
                self.assistant_message = {"role": "assistant", "content": c}
                self.raw = None

        class _MockBackend:
            async def complete(self, messages, tools, response_format=None):
                return _MockResponse(content)

        return _MockBackend()

    def _make_failing_backend(self):
        class _FailingBackend:
            async def complete(self, messages, tools, response_format=None):
                raise RuntimeError("simulated LLM error")

        return _FailingBackend()

    async def test_resolve_returns_valid_rules(self):
        import json
        from graph_pipeline.context_store import DatasetNodeType, DatasetRelationshipType
        from graph_pipeline.schema_discovery import _resolve_ambiguous_field_rules

        response_json = json.dumps({"rules": [
            {"field": "linkedItems", "delimiter": ",", "rel_type": "COVERS",
             "from_type": "", "to_type": "Requirement", "direction": "out"},
        ]})
        backend = self._make_backend(response_json)
        node_types = [DatasetNodeType(name="Requirement", maps_to="Requirement")]
        rel_types = [DatasetRelationshipType(name="COVERS", maps_to="COVERS")]

        rules = await _resolve_ambiguous_field_rules(
            [("linkedItems", ",")],
            sample=[],
            node_types=node_types,
            relationship_types=rel_types,
            id_field="uniqueId",
            backend=backend,
            max_retries=1,
        )

        assert len(rules) == 1
        assert rules[0].field == "linkedItems"
        assert rules[0].rel_type == "COVERS"

    async def test_resolve_returns_empty_on_failure(self):
        from graph_pipeline.context_store import DatasetNodeType, DatasetRelationshipType
        from graph_pipeline.schema_discovery import _resolve_ambiguous_field_rules

        backend = self._make_failing_backend()
        rules = await _resolve_ambiguous_field_rules(
            [("linkedItems", ",")],
            sample=[],
            node_types=[DatasetNodeType(name="Item", maps_to="Item")],
            relationship_types=[DatasetRelationshipType(name="LINKS", maps_to="LINKS")],
            id_field="uniqueId",
            backend=backend,
            max_retries=1,
        )
        assert rules == []


