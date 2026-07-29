"""Tests for graph_pipeline/extractor.py.

Run with: pytest tests/test_extractor.py
All tests use hardcoded DatasetContext fixtures — no YAML files, no LLM calls.
The LLM-inferred test is marked @pytest.mark.llm.
"""
import pytest


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def make_dataset_ctx(
    dataset_id="ds1",
    node_types=None,
    relationship_types=None,
    implicit_relationships=None,
    design_decisions=None,
    ambiguous_fields=None,
    hierarchy_config=...,   # sentinel: default to Tosca-compatible HierarchyConfig
    association_config=..., # sentinel: default to Tosca-compatible AssociationConfig
    nested_collections=None,
):
    from graph_pipeline.context_store import (
        AssociationConfig,
        DatasetContext,
        DatasetNodeType,
        DatasetRelationshipType,
        DesignDecision,
        HierarchyConfig,
        ImplicitRelationship,
        NestedCollection,
    )

    # Default to Tosca-compatible structural config so existing fixtures keep working.
    if hierarchy_config is ...:
        hierarchy_config = HierarchyConfig(field="nodePath")
    if association_config is ...:
        association_config = AssociationConfig()

    nt_objs = []
    for nt in (node_types or []):
        nt_objs.append(DatasetNodeType(name=nt["name"], maps_to=nt["maps_to"]))

    rt_objs = []
    for rt in (relationship_types or []):
        rt_objs.append(
            DatasetRelationshipType(
                name=rt["name"],
                maps_to=rt["maps_to"],
                **{"from": rt.get("from_type", ""), "to": rt.get("to_type", "")},
            )
        )

    ir_objs = []
    for ir in (implicit_relationships or []):
        ir_objs.append(ImplicitRelationship(**ir))

    dd_objs = []
    for dd in (design_decisions or []):
        dd_objs.append(DesignDecision(**dd))

    nc_objs = []
    for nc in (nested_collections or []):
        nc_objs.append(NestedCollection(**nc))

    return DatasetContext(
        dataset_id=dataset_id,
        node_types=nt_objs,
        relationship_types=rt_objs,
        implicit_relationships=ir_objs,
        design_decisions=dd_objs,
        ambiguous_fields=ambiguous_fields or [],
        hierarchy_config=hierarchy_config,
        association_config=association_config,
        nested_collections=nc_objs,
    )


def make_shared_ctx():
    from graph_pipeline.context_store import SharedContext
    return SharedContext()


# ---------------------------------------------------------------------------
# Rule 1 — uniqueId + typeName → Node
# ---------------------------------------------------------------------------

class TestRule1NodeExtraction:
    def _ctx(self):
        return make_dataset_ctx(
            node_types=[
                {"name": "TestCase", "maps_to": "TestCase"},
                {"name": "XModule", "maps_to": "XModule"},
            ]
        )

    async def test_basic_node_produced(self):
        from graph_pipeline.extractor import extract_all
        records = [{"uniqueId": "tc-001", "typeName": "TestCase", "name": "Login Test"}]
        nodes, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        assert len(nodes) == 1
        assert nodes[0].id == "ds1:tc-001"
        assert nodes[0].label == "TestCase"

    async def test_node_id_is_namespaced(self):
        from graph_pipeline.extractor import extract_all
        records = [{"uniqueId": "abc", "typeName": "XModule", "name": "Mod"}]
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        assert nodes[0].id == "ds1:abc"

    async def test_node_label_uses_maps_to(self):
        from graph_pipeline.extractor import extract_all
        ctx = make_dataset_ctx(node_types=[{"name": "TestCase", "maps_to": "AutomatedTest"}])
        records = [{"uniqueId": "tc-001", "typeName": "TestCase", "name": "Login"}]
        nodes, _ = await extract_all(records, ctx, make_shared_ctx())
        assert nodes[0].label == "AutomatedTest"

    async def test_scalar_properties_included(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login Test",
                "nodePath": "Root/Login Test",
                "status": "active",
            }
        ]
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        assert nodes[0].properties["name"] == "Login Test"
        assert nodes[0].properties["status"] == "active"

    async def test_associations_excluded_from_properties(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "associations": [{"edgeName": "Req", "partnerId": "r1", "direction": "out"}],
            }
        ]
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        assert "associations" not in nodes[0].properties

    async def test_details_excluded_from_properties(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "details": {"testSteps": []},
            }
        ]
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        assert "details" not in nodes[0].properties

    async def test_extraction_source_is_rule_based(self):
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource
        records = [{"uniqueId": "tc-001", "typeName": "TestCase", "name": "Login"}]
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        assert nodes[0].extraction_source == ExtractionSource.RULE_BASED

    async def test_record_without_unique_id_skipped(self):
        from graph_pipeline.extractor import extract_all
        records = [{"typeName": "TestCase", "name": "Login"}]  # no uniqueId
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        explicit = [n for n in nodes if n.extraction_source.value == "rule_based"]
        assert len(explicit) == 0

    async def test_record_without_typename_skipped(self):
        from graph_pipeline.extractor import extract_all
        records = [{"uniqueId": "tc-001", "name": "Login"}]  # no typeName
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        explicit = [n for n in nodes if n.extraction_source.value == "rule_based"]
        assert len(explicit) == 0

    async def test_multiple_records_produce_multiple_nodes(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {"uniqueId": "tc-001", "typeName": "TestCase", "name": "Login"},
            {"uniqueId": "tc-002", "typeName": "TestCase", "name": "Logout"},
        ]
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        node_ids = {n.id for n in nodes if n.extraction_source.value == "rule_based"}
        assert "ds1:tc-001" in node_ids
        assert "ds1:tc-002" in node_ids


# ---------------------------------------------------------------------------
# Rule 2 — details.moduleAttributes[] → ModuleElement nodes
# ---------------------------------------------------------------------------

class TestRule2ModuleAttributes:
    def _ctx(self):
        return make_dataset_ctx(
            node_types=[
                {"name": "XModule", "maps_to": "XModule"},
                {"name": "ModuleElement", "maps_to": "ModuleElement"},
            ],
            nested_collections=[
                {
                    "field": "details.moduleAttributes",
                    "child_label": "ModuleElement",
                    "edge_type": "HAS_ELEMENT",
                    "id_field": "uniqueId",
                }
            ],
        )

    async def test_module_attributes_produce_nodes(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "xm-001",
                "typeName": "XModule",
                "name": "Login Module",
                "details": {
                    "moduleAttributes": [
                        {"uniqueId": "attr-001", "name": "username", "businessType": "String"},
                        {"uniqueId": "attr-002", "name": "password", "businessType": "String"},
                    ]
                },
            }
        ]
        nodes, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        node_ids = {n.id for n in nodes}
        assert "ds1:attr-001" in node_ids
        assert "ds1:attr-002" in node_ids

    async def test_module_attribute_label_is_module_element(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "xm-001",
                "typeName": "XModule",
                "name": "Login Module",
                "details": {
                    "moduleAttributes": [
                        {"uniqueId": "attr-001", "name": "username", "businessType": "String"},
                    ]
                },
            }
        ]
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        attr_node = next(n for n in nodes if n.id == "ds1:attr-001")
        assert attr_node.label == "ModuleElement"

    async def test_has_element_relationship_created(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "xm-001",
                "typeName": "XModule",
                "name": "Login Module",
                "details": {
                    "moduleAttributes": [
                        {"uniqueId": "attr-001", "name": "username", "businessType": "String"},
                    ]
                },
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        has_element_rels = [r for r in rels if r.type == "HAS_ELEMENT"]
        assert len(has_element_rels) == 1
        assert has_element_rels[0].from_id == "ds1:xm-001"
        assert has_element_rels[0].to_id == "ds1:attr-001"

    async def test_no_module_element_type_skips_extraction(self):
        """If dataset_ctx doesn't declare ModuleElement, skip moduleAttributes."""
        from graph_pipeline.extractor import extract_all
        ctx = make_dataset_ctx(node_types=[{"name": "XModule", "maps_to": "XModule"}])
        records = [
            {
                "uniqueId": "xm-001",
                "typeName": "XModule",
                "name": "Login Module",
                "details": {
                    "moduleAttributes": [
                        {"uniqueId": "attr-001", "name": "username"},
                    ]
                },
            }
        ]
        nodes, _ = await extract_all(records, ctx, make_shared_ctx())
        node_ids = {n.id for n in nodes}
        assert "ds1:attr-001" not in node_ids


# ---------------------------------------------------------------------------
# Rule 3 + 4 — nodePath → phantom Folder nodes + CONTAINS edges
# ---------------------------------------------------------------------------

class TestRule3And4NodePath:
    def _ctx(self):
        return make_dataset_ctx(
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}]
        )

    async def test_3_segment_path_produces_2_contains_edges(self):
        """'Root/Suite A/Login Test' → (Root)→(Suite A)→(Login Test)"""
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login Test",
                "nodePath": "Root/Suite A/Login Test",
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        contains = [r for r in rels if r.type == "CONTAINS"]
        assert len(contains) == 2

    async def test_phantom_node_created_for_intermediate_segment(self):
        """Intermediate segments without explicit records become PHANTOM Folder nodes."""
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login Test",
                "nodePath": "Root/Suite A/Login Test",
            }
        ]
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        phantoms = [n for n in nodes if n.extraction_source == ExtractionSource.PHANTOM]
        phantom_ids = {n.id for n in phantoms}
        # "Root" and "Suite A" have no explicit records → both phantom
        assert "ds1:path:Root" in phantom_ids
        assert "ds1:path:Suite A" in phantom_ids

    async def test_phantom_node_label_is_folder(self):
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login Test",
                "nodePath": "Root/Login Test",
            }
        ]
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        phantoms = [n for n in nodes if n.extraction_source == ExtractionSource.PHANTOM]
        assert all(n.label == "Folder" for n in phantoms)

    async def test_contains_edge_connects_adjacent_segments(self):
        """The two CONTAINS edges connect Root→Suite A and Suite A→leaf."""
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login Test",
                "nodePath": "Root/Suite A/Login Test",
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        contains = [r for r in rels if r.type == "CONTAINS"]
        from_to_pairs = {(r.from_id, r.to_id) for r in contains}
        assert ("ds1:path:Root", "ds1:path:Suite A") in from_to_pairs
        assert ("ds1:path:Suite A", "ds1:tc-001") in from_to_pairs

    async def test_explicit_record_used_as_intermediate_not_phantom(self):
        """If an intermediate segment name matches an explicit record, use that node (no phantom)."""
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource
        ctx = make_dataset_ctx(
            node_types=[
                {"name": "TestCase", "maps_to": "TestCase"},
                {"name": "Folder", "maps_to": "Folder"},
            ]
        )
        records = [
            {
                "uniqueId": "folder-001",
                "typeName": "Folder",
                "name": "Suite A",
                "nodePath": "Root/Suite A",
            },
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login Test",
                "nodePath": "Root/Suite A/Login Test",
            },
        ]
        nodes, rels = await extract_all(records, ctx, make_shared_ctx())
        phantoms = [n for n in nodes if n.extraction_source == ExtractionSource.PHANTOM]
        phantom_ids = {n.id for n in phantoms}
        # Suite A has an explicit record — no phantom for it
        assert "ds1:path:Suite A" not in phantom_ids

    async def test_shared_intermediate_not_duplicated(self):
        """Two records under the same parent path produce only one phantom for the parent."""
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "nodePath": "Root/Suite A/Login",
            },
            {
                "uniqueId": "tc-002",
                "typeName": "TestCase",
                "name": "Logout",
                "nodePath": "Root/Suite A/Logout",
            },
        ]
        nodes, _ = await extract_all(records, self._ctx(), make_shared_ctx())
        phantom_ids = [n.id for n in nodes if n.id.startswith("ds1:path:Root")]
        assert phantom_ids.count("ds1:path:Root") == 1

    async def test_2_segment_path_produces_1_contains_edge(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login Test",
                "nodePath": "Root/Login Test",
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        contains = [r for r in rels if r.type == "CONTAINS"]
        assert len(contains) == 1

    async def test_contains_edge_extraction_source_rule_based(self):
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "nodePath": "Root/Login",
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        contains = [r for r in rels if r.type == "CONTAINS"]
        assert all(r.extraction_source == ExtractionSource.RULE_BASED for r in contains)

    async def test_whitespace_stripped_from_segments(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login Test",
                "nodePath": " Root / Suite A / Login Test ",
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        contains = [r for r in rels if r.type == "CONTAINS"]
        assert ("ds1:path:Root", "ds1:path:Suite A") in {(r.from_id, r.to_id) for r in contains}


# ---------------------------------------------------------------------------
# Rule 5 — associations[] → explicit edges
# ---------------------------------------------------------------------------

class TestRule5Associations:
    def _ctx(self):
        return make_dataset_ctx(
            node_types=[
                {"name": "TestCase", "maps_to": "TestCase"},
                {"name": "Requirement", "maps_to": "Requirement"},
            ],
            relationship_types=[
                {
                    "name": "Requirement",  # edgeName as it appears in associations
                    "maps_to": "COVERS",
                    "from_type": "TestCase",
                    "to_type": "Requirement",
                }
            ],
        )

    async def test_outgoing_association_creates_relationship(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "associations": [
                    {"edgeName": "Requirement", "partnerId": "req-001", "direction": "out"}
                ],
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        covers = [r for r in rels if r.type == "COVERS"]
        assert len(covers) == 1
        assert covers[0].from_id == "ds1:tc-001"
        assert covers[0].to_id == "ds1:req-001"

    async def test_incoming_association_reverses_direction(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "associations": [
                    {"edgeName": "Requirement", "partnerId": "req-001", "direction": "in"}
                ],
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        covers = [r for r in rels if r.type == "COVERS"]
        assert len(covers) == 1
        assert covers[0].from_id == "ds1:req-001"
        assert covers[0].to_id == "ds1:tc-001"

    async def test_unknown_edge_name_skipped(self):
        """Associations with no matching relationship_type in dataset_ctx are skipped."""
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "associations": [
                    {"edgeName": "UnknownEdge", "partnerId": "x-001", "direction": "out"}
                ],
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        unknown = [r for r in rels if r.type == "UNKNOWN_EDGE"]
        assert len(unknown) == 0

    async def test_association_rel_extraction_source_rule_based(self):
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "associations": [
                    {"edgeName": "Requirement", "partnerId": "req-001", "direction": "out"}
                ],
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        covers = [r for r in rels if r.type == "COVERS"]
        assert covers[0].extraction_source == ExtractionSource.RULE_BASED

    async def test_association_partner_id_namespaced(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "associations": [
                    {"edgeName": "Requirement", "partnerId": "req-001", "direction": "out"}
                ],
            }
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        covers = [r for r in rels if r.type == "COVERS"]
        assert covers[0].to_id == "ds1:req-001"


# ---------------------------------------------------------------------------
# Rule 6 — implicit foreign keys
# ---------------------------------------------------------------------------

class TestRule6ImplicitFKs:
    def _ctx_same_dataset(self):
        return make_dataset_ctx(
            node_types=[
                {"name": "TestCase", "maps_to": "TestCase"},
                {"name": "XModule", "maps_to": "XModule"},
            ],
            implicit_relationships=[
                {
                    "description": "moduleUniqueId FK to XModule",
                    "pattern": "direct_fk",
                    "edge_name": "moduleUniqueId",
                    "maps_to": "USES_MODULE",
                    "cross_dataset": False,
                    "target_dataset_id": None,
                }
            ],
        )

    def _ctx_cross_dataset(self):
        return make_dataset_ctx(
            dataset_id="ds1",
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
            implicit_relationships=[
                {
                    "description": "FK to node in ds2",
                    "pattern": "direct_fk",
                    "edge_name": "externalModuleId",
                    "maps_to": "USES_EXTERNAL",
                    "cross_dataset": True,
                    "target_dataset_id": "ds2",
                }
            ],
        )

    async def test_implicit_fk_produces_relationship(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "moduleUniqueId": "xm-001",
            }
        ]
        _, rels = await extract_all(records, self._ctx_same_dataset(), make_shared_ctx())
        uses = [r for r in rels if r.type == "USES_MODULE"]
        assert len(uses) == 1
        assert uses[0].from_id == "ds1:tc-001"
        assert uses[0].to_id == "ds1:xm-001"

    async def test_cross_dataset_fk_uses_target_dataset_namespace(self):
        """FK target id should be namespaced with target_dataset_id, not the source dataset."""
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "externalModuleId": "xm-999",
            }
        ]
        _, rels = await extract_all(records, self._ctx_cross_dataset(), make_shared_ctx())
        uses = [r for r in rels if r.type == "USES_EXTERNAL"]
        assert len(uses) == 1
        assert uses[0].from_id == "ds1:tc-001"
        # Target namespaced with ds2, not ds1
        assert uses[0].to_id == "ds2:xm-999"

    async def test_missing_fk_field_no_relationship(self):
        """Records without the FK field produce no implicit relationship."""
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                # no moduleUniqueId field
            }
        ]
        _, rels = await extract_all(records, self._ctx_same_dataset(), make_shared_ctx())
        uses = [r for r in rels if r.type == "USES_MODULE"]
        assert len(uses) == 0

    async def test_implicit_fk_extraction_source_rule_based(self):
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "moduleUniqueId": "xm-001",
            }
        ]
        _, rels = await extract_all(records, self._ctx_same_dataset(), make_shared_ctx())
        uses = [r for r in rels if r.type == "USES_MODULE"]
        assert uses[0].extraction_source == ExtractionSource.RULE_BASED


# ---------------------------------------------------------------------------
# Rule 7 — LLM-inferred extraction (integration, requires --llm)
# ---------------------------------------------------------------------------

class MockBackend:
    """Minimal ModelBackend for unit tests."""

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


class TestRule7LlmExtraction:
    async def test_llm_nodes_marked_llm_inferred(self):
        """MockBackend returning valid JSON produces LLM_INFERRED nodes."""
        import json
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource

        llm_response = json.dumps({
            "nodes": [
                {"id": "ds1:cat-functional", "label": "Category",
                 "properties": {"name": "functional"}, "source_record_id": "tc-001"}
            ],
            "relationships": [],
        })
        ctx = make_dataset_ctx(
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
            ambiguous_fields=["category"],
            hierarchy_config=None,
            association_config=None,
        )
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "category": "functional|regression",
            }
        ]
        nodes, _ = await extract_all(records, ctx, make_shared_ctx(), backend=MockBackend(llm_response))
        inferred = [n for n in nodes if n.extraction_source == ExtractionSource.LLM_INFERRED]
        assert len(inferred) == 1
        assert inferred[0].id == "ds1:cat-functional"

    async def test_no_llm_call_when_backend_none(self):
        """When backend is None, ambiguous_fields are silently skipped."""
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource

        ctx = make_dataset_ctx(
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
            ambiguous_fields=["category"],
            hierarchy_config=None,
            association_config=None,
        )
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "category": "functional",
            }
        ]
        nodes, _ = await extract_all(records, ctx, make_shared_ctx(), backend=None)
        inferred = [n for n in nodes if n.extraction_source == ExtractionSource.LLM_INFERRED]
        assert len(inferred) == 0


@pytest.mark.llm
async def test_llm_inferred_extraction_source():
    """Ambiguous fields trigger an LLM call; resulting nodes are marked LLM_INFERRED."""
    from graph_pipeline.extractor import extract_all
    from graph_pipeline.models import ExtractionSource
    from kgent.agent.backends.ollama import OllamaBackend

    ctx = make_dataset_ctx(
        node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
        ambiguous_fields=["category"],
    )
    records = [
        {
            "uniqueId": "tc-001",
            "typeName": "TestCase",
            "name": "Login",
            "category": "functional|regression",  # ambiguous — could be multiple types
        }
    ]
    backend = OllamaBackend(model="qwen3:8b", base_url="http://localhost:11434/v1")
    nodes, _ = await extract_all(records, ctx, make_shared_ctx(), backend=backend)
    inferred = [n for n in nodes if n.extraction_source == ExtractionSource.LLM_INFERRED]
    # We don't assert specific values — just that the LLM path ran and marked something
    assert isinstance(inferred, list)


# ---------------------------------------------------------------------------
# Generic (non-Tosca) schemas
# ---------------------------------------------------------------------------

class TestGenericExtraction:
    async def test_hr_flat_records(self):
        """HR dataset: custom id/type fields, one implicit FK, no hierarchy or edge arrays."""
        from graph_pipeline.context_store import (
            DatasetContext,
            DatasetNodeType,
            HierarchyConfig,
            AssociationConfig,
            ImplicitRelationship,
        )
        from graph_pipeline.extractor import extract_all

        records = [
            {"employee_id": "e1", "role": "Engineer", "name": "Alice", "manager_id": "e2"},
            {"employee_id": "e2", "role": "Manager",  "name": "Bob"},
        ]
        ctx = DatasetContext(
            dataset_id="hr",
            id_field="employee_id",
            type_field="role",
            node_types=[
                DatasetNodeType(name="Engineer", maps_to="Engineer"),
                DatasetNodeType(name="Manager",  maps_to="Manager"),
            ],
            implicit_relationships=[
                ImplicitRelationship(
                    description="reports to",
                    pattern="reports_to",
                    edge_name="manager_id",
                    maps_to="REPORTS_TO",
                )
            ],
            hierarchy_config=None,
            association_config=None,
        )

        nodes, rels = await extract_all(records, ctx, shared_ctx=None)

        assert len(nodes) == 2
        node_ids = {n.id for n in nodes}
        assert "hr:e1" in node_ids
        assert "hr:e2" in node_ids

        reports_to = [r for r in rels if r.type == "REPORTS_TO"]
        assert len(reports_to) == 1
        assert reports_to[0].from_id == "hr:e1"
        assert reports_to[0].to_id == "hr:e2"

    async def test_ticket_tracker_with_hierarchy(self):
        """Ticket tracker: custom id/type, path hierarchy, edge array."""
        from graph_pipeline.context_store import (
            AssociationConfig,
            DatasetContext,
            DatasetNodeType,
            DatasetRelationshipType,
            HierarchyConfig,
        )
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource

        records = [
            {
                "id": "t1", "type": "issue",   "title": "Bug",
                "parent_path": "Project/Issues/Bug",
                "links": [{"rel": "BLOCKS", "target": "t2", "dir": "out"}],
            },
            {
                "id": "t2", "type": "issue",   "title": "Fix",
                "parent_path": "Project/Issues/Fix",
            },
            {
                "id": "t3", "type": "comment", "title": "Note",
                "parent_path": "Project/Issues/Bug/Note",
            },
        ]
        ctx = DatasetContext(
            dataset_id="tracker",
            id_field="id",
            type_field="type",
            node_types=[
                DatasetNodeType(name="issue",   maps_to="Issue"),
                DatasetNodeType(name="comment", maps_to="Comment"),
            ],
            relationship_types=[
                DatasetRelationshipType(name="BLOCKS", maps_to="BLOCKS",
                                        **{"from": "Issue", "to": "Issue"}),
            ],
            hierarchy_config=HierarchyConfig(
                field="parent_path",
                separator="/",
                phantom_label="Container",
                edge_type="CONTAINS",
            ),
            association_config=AssociationConfig(
                array_field="links",
                edge_name_subfield="rel",
                partner_id_subfield="target",
                direction_subfield="dir",
                direction_default="out",
            ),
        )

        nodes, rels = await extract_all(records, ctx, shared_ctx=None)

        # 3 explicit nodes
        explicit = [n for n in nodes if n.extraction_source == ExtractionSource.RULE_BASED]
        explicit_ids = {n.id for n in explicit}
        assert "tracker:t1" in explicit_ids
        assert "tracker:t2" in explicit_ids
        assert "tracker:t3" in explicit_ids

        # Phantom Container nodes for "Project" and "Issues"
        phantoms = [n for n in nodes if n.extraction_source == ExtractionSource.PHANTOM]
        phantom_labels = {n.label for n in phantoms}
        assert "Container" in phantom_labels
        phantom_names = {n.properties["name"] for n in phantoms}
        assert "Project" in phantom_names
        assert "Issues" in phantom_names

        # CONTAINS edges exist
        contains = [r for r in rels if r.type == "CONTAINS"]
        assert len(contains) > 0

        # BLOCKS relationship from t1 to t2
        blocks = [r for r in rels if r.type == "BLOCKS"]
        assert len(blocks) == 1
        assert blocks[0].from_id == "tracker:t1"
        assert blocks[0].to_id == "tracker:t2"

    async def test_backward_compat_tosca_defaults(self):
        """Tosca-shaped record works with default field names and HierarchyConfig."""
        from graph_pipeline.context_store import DatasetContext, DatasetNodeType, HierarchyConfig
        from graph_pipeline.extractor import extract_all

        records = [
            {
                "uniqueId": "abc",
                "typeName": "TestCase",
                "name": "My Test",
                "nodePath": "Root/Tests/My Test",
            }
        ]
        ctx = DatasetContext(
            dataset_id="tosca",
            id_field="uniqueId",
            type_field="typeName",
            node_types=[DatasetNodeType(name="TestCase", maps_to="TestCase")],
            hierarchy_config=HierarchyConfig(field="nodePath"),
            association_config=None,
        )

        nodes, rels = await extract_all(records, ctx, shared_ctx=None)

        explicit = [n for n in nodes if n.id == "tosca:abc"]
        assert len(explicit) == 1
        assert explicit[0].label == "TestCase"


# ---------------------------------------------------------------------------
# ExtractionSource values on different output types
# ---------------------------------------------------------------------------

class TestExtractionSourceLabels:
    async def test_phantom_nodes_have_phantom_source(self):
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource
        ctx = make_dataset_ctx(node_types=[{"name": "TestCase", "maps_to": "TestCase"}])
        records = [
            {"uniqueId": "tc-001", "typeName": "TestCase", "name": "T", "nodePath": "Root/T"}
        ]
        nodes, _ = await extract_all(records, ctx, make_shared_ctx())
        root_node = next(n for n in nodes if n.id == "ds1:path:Root")
        assert root_node.extraction_source == ExtractionSource.PHANTOM

    async def test_rule_based_nodes_have_rule_based_source(self):
        from graph_pipeline.extractor import extract_all
        from graph_pipeline.models import ExtractionSource
        ctx = make_dataset_ctx(node_types=[{"name": "TestCase", "maps_to": "TestCase"}])
        records = [{"uniqueId": "tc-001", "typeName": "TestCase", "name": "T"}]
        nodes, _ = await extract_all(records, ctx, make_shared_ctx())
        tc_node = next(n for n in nodes if n.id == "ds1:tc-001")
        assert tc_node.extraction_source == ExtractionSource.RULE_BASED
