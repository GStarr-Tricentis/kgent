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
    ambiguous_field_rules=None,
    hierarchy_config=...,   # sentinel: default to Tosca-compatible HierarchyConfig
    association_config=..., # sentinel: default to Tosca-compatible AssociationConfig
    nested_collections=None,
    property_paths=None,
    path_fk_relationships=None,
):
    from graph_pipeline.context_store import (
        AmbiguousFieldRule,
        AssociationConfig,
        DatasetContext,
        DatasetNodeType,
        DatasetRelationshipType,
        DesignDecision,
        HierarchyConfig,
        ImplicitRelationship,
        NestedCollection,
        PathFKRelationship,
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

    pfk_objs = []
    for pfk in (path_fk_relationships or []):
        pfk_objs.append(PathFKRelationship(**pfk))

    afr_objs = []
    for afr in (ambiguous_field_rules or []):
        afr_objs.append(AmbiguousFieldRule(**afr))

    return DatasetContext(
        dataset_id=dataset_id,
        node_types=nt_objs,
        relationship_types=rt_objs,
        implicit_relationships=ir_objs,
        design_decisions=dd_objs,
        ambiguous_fields=ambiguous_fields or [],
        ambiguous_field_rules=afr_objs,
        hierarchy_config=hierarchy_config,
        association_config=association_config,
        nested_collections=nc_objs,
        property_paths=property_paths or [],
        path_fk_relationships=pfk_objs,
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

class TestApplyAmbiguousFieldRules:
    """Unit tests for _apply_ambiguous_field_rules."""

    def _rule(self, field="refs", delimiter=",", rel_type="LINKS_TO",
               from_type="", to_type="Item", direction="out"):
        from graph_pipeline.context_store import AmbiguousFieldRule
        return AmbiguousFieldRule(
            field=field, delimiter=delimiter, rel_type=rel_type,
            from_type=from_type, to_type=to_type, direction=direction,
        )

    def test_delimited_tokens_matching_uid_set_produce_rels(self):
        from graph_pipeline.extractor import _apply_ambiguous_field_rules
        record = {"uniqueId": "tc-1", "typeName": "TestCase", "refs": "tc-2,tc-3,tc-99"}
        uid_set = {"tc-1", "tc-2", "tc-3"}
        rels = _apply_ambiguous_field_rules(
            record, "ds", "tc-1", "TestCase", [self._rule()], uid_set
        )
        assert len(rels) == 2
        to_ids = {r.to_id for r in rels}
        assert to_ids == {"ds:tc-2", "ds:tc-3"}

    def test_token_not_in_uid_set_is_skipped(self):
        from graph_pipeline.extractor import _apply_ambiguous_field_rules
        record = {"uniqueId": "tc-1", "typeName": "TestCase", "refs": "tc-2,unknown-99"}
        uid_set = {"tc-1", "tc-2"}
        rels = _apply_ambiguous_field_rules(
            record, "ds", "tc-1", "TestCase", [self._rule()], uid_set
        )
        assert len(rels) == 1
        assert rels[0].to_id == "ds:tc-2"

    def test_whole_value_match_when_delimiter_empty(self):
        from graph_pipeline.extractor import _apply_ambiguous_field_rules
        record = {"uniqueId": "tc-1", "typeName": "TestCase", "refs": "tc-2"}
        uid_set = {"tc-1", "tc-2"}
        rels = _apply_ambiguous_field_rules(
            record, "ds", "tc-1", "TestCase", [self._rule(delimiter="")], uid_set
        )
        assert len(rels) == 1
        assert rels[0].to_id == "ds:tc-2"

    def test_direction_in_reverses_from_to(self):
        from graph_pipeline.extractor import _apply_ambiguous_field_rules
        record = {"uniqueId": "tc-1", "typeName": "TestCase", "refs": "tc-2"}
        uid_set = {"tc-1", "tc-2"}
        rels = _apply_ambiguous_field_rules(
            record, "ds", "tc-1", "TestCase",
            [self._rule(direction="in")], uid_set,
        )
        assert len(rels) == 1
        assert rels[0].from_id == "ds:tc-2"
        assert rels[0].to_id == "ds:tc-1"

    def test_from_type_empty_uses_this_label(self):
        from graph_pipeline.extractor import _apply_ambiguous_field_rules
        record = {"uniqueId": "tc-1", "typeName": "TestCase", "refs": "tc-2"}
        uid_set = {"tc-1", "tc-2"}
        rels = _apply_ambiguous_field_rules(
            record, "ds", "tc-1", "MyLabel", [self._rule(from_type="")], uid_set
        )
        assert rels[0].from_label == "MyLabel"

    def test_from_type_set_overrides_this_label(self):
        from graph_pipeline.extractor import _apply_ambiguous_field_rules
        record = {"uniqueId": "tc-1", "typeName": "TestCase", "refs": "tc-2"}
        uid_set = {"tc-1", "tc-2"}
        rels = _apply_ambiguous_field_rules(
            record, "ds", "tc-1", "MyLabel", [self._rule(from_type="Override")], uid_set
        )
        assert rels[0].from_label == "Override"

    def test_missing_field_skipped(self):
        from graph_pipeline.extractor import _apply_ambiguous_field_rules
        record = {"uniqueId": "tc-1", "typeName": "TestCase"}
        uid_set = {"tc-1", "tc-2"}
        rels = _apply_ambiguous_field_rules(
            record, "ds", "tc-1", "TestCase", [self._rule()], uid_set
        )
        assert rels == []

    def test_empty_uid_set_produces_no_rels(self):
        from graph_pipeline.extractor import _apply_ambiguous_field_rules
        record = {"uniqueId": "tc-1", "typeName": "TestCase", "refs": "tc-2,tc-3"}
        rels = _apply_ambiguous_field_rules(
            record, "ds", "tc-1", "TestCase", [self._rule()], set()
        )
        assert rels == []

    def test_comma_delimited_field(self):
        from graph_pipeline.context_store import AmbiguousFieldRule
        from graph_pipeline.extractor import _apply_ambiguous_field_rules
        rule = AmbiguousFieldRule(
            field="relatedIds", delimiter=",", rel_type="COVERS",
            from_type="", to_type="Requirement", direction="out",
        )
        record = {"uniqueId": "tc1", "relatedIds": "req1,req2,unknown"}
        uid_set = {"req1", "req2"}
        rels = _apply_ambiguous_field_rules(record, "ds", "tc1", "TestCase", [rule], uid_set)
        assert len(rels) == 2
        assert all(r.from_id == "ds:tc1" for r in rels)
        assert {r.to_id for r in rels} == {"ds:req1", "ds:req2"}
        assert all(r.type == "COVERS" for r in rels)
        assert not any(r.to_id == "ds:unknown" for r in rels)


class TestRule7DeterministicExtraction:
    """Rule 7 via extract_all using ambiguous_field_rules (no LLM)."""

    def _ctx(self):
        return make_dataset_ctx(
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
            relationship_types=[{
                "name": "LINKS_TO", "maps_to": "LINKS_TO",
                "from_type": "TestCase", "to_type": "TestCase",
            }],
            ambiguous_field_rules=[{
                "field": "refs", "delimiter": ",", "rel_type": "LINKS_TO",
                "from_type": "", "to_type": "TestCase", "direction": "out",
            }],
            hierarchy_config=None,
            association_config=None,
        )

    async def test_matching_tokens_produce_rels(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {"uniqueId": "tc-1", "typeName": "TestCase", "refs": "tc-2,tc-3"},
            {"uniqueId": "tc-2", "typeName": "TestCase"},
            {"uniqueId": "tc-3", "typeName": "TestCase"},
        ]
        _, rels = await extract_all(records, self._ctx(), make_shared_ctx())
        rule_rels = [r for r in rels if r.type == "LINKS_TO"]
        assert len(rule_rels) == 2
        to_ids = {r.to_id for r in rule_rels}
        assert to_ids == {"ds1:tc-2", "ds1:tc-3"}

    async def test_no_rules_produces_no_extra_rels(self):
        from graph_pipeline.extractor import extract_all
        ctx = make_dataset_ctx(
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
            hierarchy_config=None,
            association_config=None,
        )
        records = [
            {"uniqueId": "tc-1", "typeName": "TestCase", "refs": "tc-2"},
            {"uniqueId": "tc-2", "typeName": "TestCase"},
        ]
        _, rels = await extract_all(records, ctx, make_shared_ctx())
        assert rels == []



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


# ---------------------------------------------------------------------------
# build_extraction_indices — uid_set population
# ---------------------------------------------------------------------------

class TestBuildExtractionIndicesUidSet:
    async def test_uid_set_populated_with_raw_uids(self):
        from graph_pipeline.extractor import build_extraction_indices
        ctx = make_dataset_ctx(
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
            hierarchy_config=None,
            association_config=None,
        )
        records = [
            {"uniqueId": "tc-1", "typeName": "TestCase"},
            {"uniqueId": "tc-2", "typeName": "TestCase"},
            {"uniqueId": "tc-3", "typeName": "TestCase"},
        ]
        indices = build_extraction_indices(iter(records), ctx)
        assert indices.uid_set == {"tc-1", "tc-2", "tc-3"}

    async def test_record_missing_uid_not_added(self):
        from graph_pipeline.extractor import build_extraction_indices
        ctx = make_dataset_ctx(
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
            hierarchy_config=None,
            association_config=None,
        )
        records = [
            {"uniqueId": "tc-1", "typeName": "TestCase"},
            {"typeName": "TestCase"},  # no uniqueId
        ]
        indices = build_extraction_indices(iter(records), ctx)
        assert indices.uid_set == {"tc-1"}


# (TestLlmExtractBatch and TestLlmExtractAmbiguous removed — Rule 7 is now deterministic)

# ---------------------------------------------------------------------------
# property_paths — flat nested dict merging into node properties
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# property_paths — flat nested dict merging into node properties
# ---------------------------------------------------------------------------

class TestPropertyPaths:
    def _ctx(self, property_paths, node_types=None):
        return make_dataset_ctx(
            node_types=node_types or [{"name": "Item", "maps_to": "Item"}],
            hierarchy_config=None,
            association_config=None,
            property_paths=property_paths,
        )

    async def test_flat_dict_merged_into_node_properties(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "r1",
                "typeName": "Item",
                "attributes": {"requirementType": "Requirement", "weight": "1"},
            }
        ]
        ctx = self._ctx(property_paths=["attributes"])
        nodes, _ = await extract_all(records, ctx, shared_ctx=None)
        node = next(n for n in nodes if n.source_record_id == "r1")
        assert node.properties.get("requirementType") == "Requirement"
        assert node.properties.get("weight") == "1"

    async def test_top_level_scalar_wins_on_collision(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "r1",
                "typeName": "Item",
                "name": "TopLevel",
                "attributes": {"name": "Nested"},
            }
        ]
        ctx = self._ctx(property_paths=["attributes"])
        nodes, _ = await extract_all(records, ctx, shared_ctx=None)
        node = next(n for n in nodes if n.source_record_id == "r1")
        assert node.properties["name"] == "TopLevel"

    async def test_empty_property_paths_unchanged(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "r1",
                "typeName": "Item",
                "name": "X",
                "attributes": {"businessType": "Widget"},
            }
        ]
        ctx = self._ctx(property_paths=[])
        nodes, _ = await extract_all(records, ctx, shared_ctx=None)
        node = next(n for n in nodes if n.source_record_id == "r1")
        assert "businessType" not in node.properties

    async def test_scalar_dot_path(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "r1",
                "typeName": "Item",
                "details": {"businessType": "AnyUIWindow", "kind": "module"},
            }
        ]
        ctx = self._ctx(property_paths=["details.businessType"])
        nodes, _ = await extract_all(records, ctx, shared_ctx=None)
        node = next(n for n in nodes if n.source_record_id == "r1")
        assert node.properties.get("businessType") == "AnyUIWindow"


# ---------------------------------------------------------------------------
# path_fk_relationships — path-string FK resolution (Rule 6b)
# ---------------------------------------------------------------------------

class TestPathFKRelationships:
    def _ctx(self, path_fk_relationships, node_types=None):
        return make_dataset_ctx(
            node_types=node_types or [
                {"name": "Parent", "maps_to": "Parent"},
                {"name": "Child", "maps_to": "Child"},
            ],
            hierarchy_config=None,
            association_config=None,
            path_fk_relationships=path_fk_relationships,
        )

    async def test_top_level_path_fk(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {"uniqueId": "p1", "typeName": "Parent", "childPath": "/root/child-a"},
            {"uniqueId": "c1", "typeName": "Child",  "nodePath": "/root/child-a"},
        ]
        ctx = self._ctx([{
            "container_path": None,
            "fk_field": "childPath",
            "target_field": "nodePath",
            "maps_to": "HAS_CHILD",
            "from_type": "Parent",
            "to_type": "Child",
        }])
        _, rels = await extract_all(records, ctx, shared_ctx=None)
        path_rels = [r for r in rels if r.type == "HAS_CHILD"]
        assert len(path_rels) == 1
        assert path_rels[0].from_id == "ds1:p1"
        assert path_rels[0].to_id == "ds1:c1"

    async def test_nested_container_path_fk(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {
                "uniqueId": "el1",
                "typeName": "Parent",
                "details": {
                    "entries": [
                        {"testCaseNodePath": "/root/tc-a"},
                        {"testCaseNodePath": None},
                    ]
                },
            },
            {"uniqueId": "tc1", "typeName": "Child", "nodePath": "/root/tc-a"},
        ]
        ctx = self._ctx([{
            "container_path": "details.entries",
            "fk_field": "testCaseNodePath",
            "target_field": "nodePath",
            "maps_to": "REFERENCES",
            "from_type": "Parent",
            "to_type": "Child",
        }])
        _, rels = await extract_all(records, ctx, shared_ctx=None)
        ref_rels = [r for r in rels if r.type == "REFERENCES"]
        assert len(ref_rels) == 1
        assert ref_rels[0].from_id == "ds1:el1"
        assert ref_rels[0].to_id == "ds1:tc1"

    async def test_unresolved_path_skipped(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {"uniqueId": "p1", "typeName": "Parent", "childPath": "/does/not/exist"},
        ]
        ctx = self._ctx([{
            "container_path": None,
            "fk_field": "childPath",
            "target_field": "nodePath",
            "maps_to": "HAS_CHILD",
            "from_type": "Parent",
            "to_type": "Child",
        }])
        _, rels = await extract_all(records, ctx, shared_ctx=None)
        assert not any(r.type == "HAS_CHILD" for r in rels)

    async def test_empty_path_fk_relationships(self):
        from graph_pipeline.extractor import extract_all
        records = [
            {"uniqueId": "p1", "typeName": "Parent", "nodePath": "/root/p1"},
        ]
        ctx = self._ctx([])
        nodes, rels = await extract_all(records, ctx, shared_ctx=None)
        assert len(nodes) == 1
        assert all(r.extraction_source.value != "RULE_BASED" or r.type != "HAS_CHILD"
                   for r in rels)


# ---------------------------------------------------------------------------
# build_extraction_indices (Pass 2)
# ---------------------------------------------------------------------------

class TestBuildExtractionIndices:
    def _ctx_with_pfk(self, target_field="nodePath"):
        return make_dataset_ctx(
            dataset_id="ds1",
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
            path_fk_relationships=[{
                "fk_field": "nodePath",
                "target_field": target_field,
                "maps_to": "BELONGS_TO",
                "from_type": "TestCase",
                "to_type": "Folder",
            }],
        )

    def test_name_index_populated(self):
        from graph_pipeline.extractor import build_extraction_indices
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
        )
        records = [
            {"uniqueId": "tc-1", "typeName": "TestCase", "name": "Login"},
            {"uniqueId": "tc-2", "typeName": "TestCase", "name": "Logout"},
        ]
        indices = build_extraction_indices(iter(records), ctx)
        assert "Login" in indices.name_to_node
        assert indices.name_to_node["Login"] == ("ds1:tc-1", "TestCase")
        assert "Logout" in indices.name_to_node
        assert indices.name_to_node["Logout"] == ("ds1:tc-2", "TestCase")

    def test_uid_set_populated(self):
        from graph_pipeline.extractor import build_extraction_indices
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[{"name": "Item", "maps_to": "Item"}],
            hierarchy_config=None,
            association_config=None,
        )
        records = [
            {"uniqueId": "r1", "typeName": "Item"},
            {"uniqueId": "r2", "typeName": "Item"},
        ]
        indices = build_extraction_indices(iter(records), ctx)
        assert indices.uid_set == {"r1", "r2"}

    def test_path_value_index_populated(self):
        from graph_pipeline.extractor import build_extraction_indices
        ctx = self._ctx_with_pfk(target_field="nodePath")
        records = [
            {"uniqueId": "tc-1", "typeName": "TestCase", "name": "Login", "nodePath": "foo"},
        ]
        indices = build_extraction_indices(iter(records), ctx)
        assert "nodePath" in indices.path_value_index
        assert indices.path_value_index["nodePath"]["foo"] == "ds1:tc-1"

    def test_records_without_id_or_type_skipped(self):
        from graph_pipeline.extractor import build_extraction_indices
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
        )
        records = [
            {"typeName": "TestCase", "name": "No ID"},          # missing uniqueId
            {"uniqueId": "tc-1", "name": "No Type"},            # missing typeName
            {"uniqueId": "tc-2", "typeName": "TestCase", "name": "Valid"},
        ]
        indices = build_extraction_indices(iter(records), ctx)
        assert len(indices.name_to_node) == 1
        assert "Valid" in indices.name_to_node

    def test_path_value_index_key_initialized_even_if_no_records_match(self):
        from graph_pipeline.extractor import build_extraction_indices
        ctx = self._ctx_with_pfk(target_field="nodePath")
        records = [
            {"uniqueId": "tc-1", "typeName": "TestCase", "name": "Login"},
            # no record has a "nodePath" field
        ]
        indices = build_extraction_indices(iter(records), ctx)
        assert "nodePath" in indices.path_value_index


# ---------------------------------------------------------------------------
# extract_and_write_stream — streaming extraction (Phase 5)
# ---------------------------------------------------------------------------

class _FakeWriteBuffer:
    """Captures add_node/add_rel calls without writing to Neo4j."""
    def __init__(self):
        from graph_pipeline.neo4j_writer import WriteResult
        self.nodes: list = []
        self.rels: list = []
        self.flush_count: int = 0
        self.result = WriteResult()

    async def add_node(self, node):
        self.nodes.append(node)

    async def add_rel(self, rel):
        self.rels.append(rel)

    async def flush_all(self):
        self.flush_count += 1


class TestExtractAndWriteStream:
    async def test_stream_rule1_primary_node_added_to_buffer(self):
        from graph_pipeline.extractor import ExtractionIndices, extract_and_write_stream
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
        )
        records = [{"uniqueId": "tc-001", "typeName": "TestCase", "name": "Login"}]
        buf = _FakeWriteBuffer()
        await extract_and_write_stream(iter(records), ctx, None, ExtractionIndices(), buf)
        node_ids = [n.id for n in buf.nodes]
        assert "ds1:tc-001" in node_ids
        matched = next(n for n in buf.nodes if n.id == "ds1:tc-001")
        assert matched.label == "TestCase"

    async def test_stream_rule2_nested_collection_nodes_and_rels(self):
        from graph_pipeline.extractor import ExtractionIndices, extract_and_write_stream
        ctx = make_dataset_ctx(
            dataset_id="ds1",
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
        records = [
            {
                "uniqueId": "xm-001",
                "typeName": "XModule",
                "name": "Login Module",
                "details": {
                    "moduleAttributes": [
                        {"uniqueId": "attr-001", "name": "username"},
                        {"uniqueId": "attr-002", "name": "password"},
                    ]
                },
            }
        ]
        buf = _FakeWriteBuffer()
        await extract_and_write_stream(iter(records), ctx, None, ExtractionIndices(), buf)
        node_ids = {n.id for n in buf.nodes}
        assert "ds1:attr-001" in node_ids
        assert "ds1:attr-002" in node_ids
        has_element_rels = [r for r in buf.rels if r.type == "HAS_ELEMENT"]
        assert len(has_element_rels) == 2
        assert has_element_rels[0].from_id == "ds1:xm-001"

    async def test_stream_rule5_association_edges(self):
        from graph_pipeline.extractor import ExtractionIndices, extract_and_write_stream
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[
                {"name": "TestCase", "maps_to": "TestCase"},
                {"name": "Requirement", "maps_to": "Requirement"},
            ],
            relationship_types=[
                {
                    "name": "Requirement",
                    "maps_to": "COVERS",
                    "from_type": "TestCase",
                    "to_type": "Requirement",
                }
            ],
        )
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
        buf = _FakeWriteBuffer()
        await extract_and_write_stream(iter(records), ctx, None, ExtractionIndices(), buf)
        covers = [r for r in buf.rels if r.type == "COVERS"]
        assert len(covers) == 1
        assert covers[0].from_id == "ds1:tc-001"
        assert covers[0].to_id == "ds1:req-001"

    async def test_stream_rule6_implicit_fk(self):
        from graph_pipeline.extractor import ExtractionIndices, extract_and_write_stream
        ctx = make_dataset_ctx(
            dataset_id="ds1",
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
        records = [
            {
                "uniqueId": "tc-001",
                "typeName": "TestCase",
                "name": "Login",
                "moduleUniqueId": "xm-001",
            }
        ]
        buf = _FakeWriteBuffer()
        await extract_and_write_stream(iter(records), ctx, None, ExtractionIndices(), buf)
        fk_rels = [r for r in buf.rels if r.type == "USES_MODULE"]
        assert len(fk_rels) == 1
        assert fk_rels[0].from_id == "ds1:tc-001"
        assert fk_rels[0].to_id == "ds1:xm-001"

    async def test_stream_phantom_nodes_not_duplicated(self):
        """Two records sharing the same phantom parent must emit only one phantom node."""
        from graph_pipeline.extractor import ExtractionIndices, build_extraction_indices, extract_and_write_stream
        from graph_pipeline.models import ExtractionSource
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
        )
        records = [
            {"uniqueId": "tc-1", "typeName": "TestCase", "name": "A", "nodePath": "Root/A"},
            {"uniqueId": "tc-2", "typeName": "TestCase", "name": "B", "nodePath": "Root/B"},
        ]
        # Pass 2: build name_to_node so _emit_hierarchy_inline is used in Pass 3
        indices = build_extraction_indices(iter(records), ctx)
        assert indices.name_to_node  # guard: must be non-empty to trigger inline path

        buf = _FakeWriteBuffer()
        await extract_and_write_stream(iter(records), ctx, None, indices, buf)

        phantom_ids = [n.id for n in buf.nodes if n.extraction_source == ExtractionSource.PHANTOM]
        assert phantom_ids.count("ds1:path:Root") == 1

    async def test_stream_rule7_deterministic_rels_added_to_pending(self):
        from graph_pipeline.extractor import ExtractionIndices, build_extraction_indices, extract_and_write_stream
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
            relationship_types=[{
                "name": "LINKS_TO", "maps_to": "LINKS_TO",
                "from_type": "TestCase", "to_type": "TestCase",
            }],
            ambiguous_field_rules=[{
                "field": "refs", "delimiter": ",", "rel_type": "LINKS_TO",
                "from_type": "", "to_type": "TestCase", "direction": "out",
            }],
            hierarchy_config=None,
            association_config=None,
        )
        records = [
            {"uniqueId": "tc-1", "typeName": "TestCase", "refs": "tc-2,tc-3"},
            {"uniqueId": "tc-2", "typeName": "TestCase"},
            {"uniqueId": "tc-3", "typeName": "TestCase"},
        ]
        buf = _FakeWriteBuffer()
        indices = build_extraction_indices(iter(records), ctx)
        result = await extract_and_write_stream(iter(records), ctx, None, indices, buf)
        rule_rels = [r for r in buf.rels if r.type == "LINKS_TO"]
        assert len(rule_rels) == 2
        to_ids = {r.to_id for r in rule_rels}
        assert to_ids == {"ds1:tc-2", "ds1:tc-3"}
        assert not hasattr(result, "llm_buffer")

    async def test_stream_flush_all_called_at_end_of_extraction(self):
        from graph_pipeline.extractor import ExtractionIndices, extract_and_write_stream
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[{"name": "TestCase", "maps_to": "TestCase"}],
        )
        records = [{"uniqueId": "tc-1", "typeName": "TestCase", "name": "A"}]
        buf = _FakeWriteBuffer()
        await extract_and_write_stream(iter(records), ctx, None, ExtractionIndices(), buf)
        assert buf.flush_count >= 1

    async def test_write_rels_false_produces_no_rels(self):
        """Pass 3a: write_rels=False must produce nodes but zero rels."""
        from graph_pipeline.extractor import ExtractionIndices, extract_and_write_stream
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[
                {"name": "TestCase", "maps_to": "TestCase"},
                {"name": "Requirement", "maps_to": "Requirement"},
            ],
            relationship_types=[
                {
                    "name": "Requirement",
                    "maps_to": "COVERS",
                    "from_type": "TestCase",
                    "to_type": "Requirement",
                }
            ],
        )
        record = {
            "uniqueId": "r1", "typeName": "TestCase",
            "associations": [
                {"edgeName": "Requirement", "partnerId": "r2", "direction": "out"}
            ],
        }
        buf = _FakeWriteBuffer()
        await extract_and_write_stream(
            iter([record]), ctx, None, ExtractionIndices(), buf,
            write_rels=False,
        )
        assert len(buf.nodes) > 0, "nodes should be written in Pass 3a"
        assert len(buf.rels) == 0, "rels must not be written when write_rels=False"

    async def test_write_nodes_false_produces_no_nodes(self):
        """Pass 3b: write_nodes=False must produce rels but zero nodes."""
        from graph_pipeline.extractor import ExtractionIndices, extract_and_write_stream
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[
                {"name": "TestCase", "maps_to": "TestCase"},
                {"name": "Requirement", "maps_to": "Requirement"},
            ],
            relationship_types=[
                {
                    "name": "Requirement",
                    "maps_to": "COVERS",
                    "from_type": "TestCase",
                    "to_type": "Requirement",
                }
            ],
        )
        record = {
            "uniqueId": "r1", "typeName": "TestCase",
            "associations": [
                {"edgeName": "Requirement", "partnerId": "r2", "direction": "out"}
            ],
        }
        buf = _FakeWriteBuffer()
        await extract_and_write_stream(
            iter([record]), ctx, None, ExtractionIndices(), buf,
            write_nodes=False,
        )
        assert len(buf.nodes) == 0, "nodes must not be written when write_nodes=False"
        assert len(buf.rels) > 0, "rels should be written in Pass 3b"
