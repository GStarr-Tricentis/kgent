"""Tests for graph_pipeline/cypher_generator.py.

Run with: pytest tests/test_cypher_generator.py

generate_node_merge and generate_relationship_merge return (cypher: str, params: dict).
Tests check both the Cypher string structure and the params dict values.
"""
from graph_pipeline.models import ExtractionSource, Node, Relationship


def make_node(
    id="ds1:tc-001",
    label="TestCase",
    properties=None,
    source_record_id="tc-001",
    extraction_source=ExtractionSource.RULE_BASED,
):
    return Node(
        id=id,
        label=label,
        properties=properties or {"name": "Login Test"},
        source_record_id=source_record_id,
        extraction_source=extraction_source,
    )


def make_rel(
    from_id="ds1:tc-001",
    to_id="ds1:req-001",
    from_label="TestCase",
    to_label="Requirement",
    type="COVERS",
    properties=None,
    source_record_id="tc-001",
    extraction_source=ExtractionSource.RULE_BASED,
):
    return Relationship(
        from_id=from_id,
        to_id=to_id,
        from_label=from_label,
        to_label=to_label,
        type=type,
        properties=properties or {},
        source_record_id=source_record_id,
        extraction_source=extraction_source,
    )


# ---------------------------------------------------------------------------
# generate_node_merge
# ---------------------------------------------------------------------------

class TestGenerateNodeMerge:
    def test_returns_tuple(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        result = generate_node_merge(make_node())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cypher_is_string(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        cypher, _ = generate_node_merge(make_node())
        assert isinstance(cypher, str)

    def test_params_is_dict(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        _, params = generate_node_merge(make_node())
        assert isinstance(params, dict)

    def test_contains_merge_keyword(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        cypher, _ = generate_node_merge(make_node())
        assert "MERGE" in cypher

    def test_contains_node_label(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        cypher, _ = generate_node_merge(make_node(label="TestCase"))
        assert "TestCase" in cypher

    def test_id_in_params(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        _, params = generate_node_merge(make_node(id="ds1:tc-001"))
        assert params["id"] == "ds1:tc-001"

    def test_id_placeholder_in_cypher(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        cypher, _ = generate_node_merge(make_node())
        assert "$id" in cypher

    def test_props_in_params(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        node = make_node(properties={"name": "Login Test", "status": "active"})
        _, params = generate_node_merge(node)
        assert params["props"] == {"name": "Login Test", "status": "active"}

    def test_props_placeholder_in_cypher(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        cypher, _ = generate_node_merge(make_node())
        assert "$props" in cypher
        assert "+=" in cypher

    def test_extraction_source_in_params_as_string(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        node = make_node(extraction_source=ExtractionSource.RULE_BASED)
        _, params = generate_node_merge(node)
        assert params["extraction_source"] == "rule_based"

    def test_phantom_extraction_source_in_params(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        node = make_node(extraction_source=ExtractionSource.PHANTOM)
        _, params = generate_node_merge(node)
        assert params["extraction_source"] == "phantom"

    def test_llm_inferred_extraction_source_in_params(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        node = make_node(extraction_source=ExtractionSource.LLM_INFERRED)
        _, params = generate_node_merge(node)
        assert params["extraction_source"] == "llm_inferred"

    def test_ingested_at_present(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        cypher, _ = generate_node_merge(make_node())
        assert "ingested_at" in cypher
        assert "datetime()" in cypher

    def test_different_labels_produce_different_cypher(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        cypher1, _ = generate_node_merge(make_node(label="TestCase"))
        cypher2, _ = generate_node_merge(make_node(label="Requirement"))
        assert "TestCase" in cypher1
        assert "Requirement" in cypher2
        assert cypher1 != cypher2

    def test_uses_merge_not_create(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        cypher, _ = generate_node_merge(make_node())
        assert "MERGE" in cypher
        assert "CREATE" not in cypher


# ---------------------------------------------------------------------------
# generate_relationship_merge
# ---------------------------------------------------------------------------

class TestGenerateRelationshipMerge:
    def test_returns_tuple(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        result = generate_relationship_merge(make_rel())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cypher_is_string(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        cypher, _ = generate_relationship_merge(make_rel())
        assert isinstance(cypher, str)

    def test_contains_merge_keyword(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        cypher, _ = generate_relationship_merge(make_rel())
        assert "MERGE" in cypher

    def test_from_label_in_match_clause(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        cypher, _ = generate_relationship_merge(make_rel(from_label="TestCase"))
        lines = cypher.splitlines()
        match_lines = [l for l in lines if "MATCH" in l]
        assert any("TestCase" in l for l in match_lines)

    def test_to_label_in_match_clause(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        cypher, _ = generate_relationship_merge(make_rel(to_label="Requirement"))
        lines = cypher.splitlines()
        match_lines = [l for l in lines if "MATCH" in l]
        assert any("Requirement" in l for l in match_lines)

    def test_relationship_type_present(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        cypher, _ = generate_relationship_merge(make_rel(type="COVERS"))
        assert "COVERS" in cypher

    def test_from_id_in_params(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        _, params = generate_relationship_merge(make_rel(from_id="ds1:tc-001"))
        assert params["from_id"] == "ds1:tc-001"

    def test_to_id_in_params(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        _, params = generate_relationship_merge(make_rel(to_id="ds1:req-001"))
        assert params["to_id"] == "ds1:req-001"

    def test_extraction_source_in_params_as_string(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        rel = make_rel(extraction_source=ExtractionSource.RULE_BASED)
        _, params = generate_relationship_merge(rel)
        assert params["extraction_source"] == "rule_based"

    def test_two_match_clauses(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        cypher, _ = generate_relationship_merge(make_rel())
        assert cypher.count("MATCH") == 2

    def test_uses_merge_not_create(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        cypher, _ = generate_relationship_merge(make_rel())
        assert "MERGE" in cypher


# ---------------------------------------------------------------------------
# generate_constraint_statements
# ---------------------------------------------------------------------------

class TestGenerateConstraintStatements:
    def test_returns_list(self):
        from graph_pipeline.cypher_generator import generate_constraint_statements
        result = generate_constraint_statements(["TestCase"])
        assert isinstance(result, list)

    def test_one_statement_per_label(self):
        from graph_pipeline.cypher_generator import generate_constraint_statements
        result = generate_constraint_statements(["TestCase", "Requirement", "Folder"])
        assert len(result) == 3

    def test_empty_labels_returns_empty(self):
        from graph_pipeline.cypher_generator import generate_constraint_statements
        assert generate_constraint_statements([]) == []

    def test_uses_create_constraint_not_index(self):
        from graph_pipeline.cypher_generator import generate_constraint_statements
        for stmt in generate_constraint_statements(["TestCase"]):
            assert "CREATE CONSTRAINT" in stmt
            assert "CREATE INDEX" not in stmt

    def test_uses_require_is_unique(self):
        from graph_pipeline.cypher_generator import generate_constraint_statements
        for stmt in generate_constraint_statements(["TestCase"]):
            assert "REQUIRE" in stmt
            assert "IS UNIQUE" in stmt

    def test_uses_if_not_exists(self):
        from graph_pipeline.cypher_generator import generate_constraint_statements
        for stmt in generate_constraint_statements(["TestCase"]):
            assert "IF NOT EXISTS" in stmt

    def test_label_appears_in_statement(self):
        from graph_pipeline.cypher_generator import generate_constraint_statements
        stmts = generate_constraint_statements(["Requirement"])
        assert any("Requirement" in s for s in stmts)

    def test_id_property_constrained(self):
        from graph_pipeline.cypher_generator import generate_constraint_statements
        for stmt in generate_constraint_statements(["TestCase"]):
            assert ".id" in stmt or "n.id" in stmt


# ---------------------------------------------------------------------------
# generate_extraction_source_index_statements
# ---------------------------------------------------------------------------

class TestGenerateExtractionSourceIndexStatements:
    def test_returns_list(self):
        from graph_pipeline.cypher_generator import generate_extraction_source_index_statements
        result = generate_extraction_source_index_statements(["TestCase"])
        assert isinstance(result, list)

    def test_one_statement_per_label(self):
        from graph_pipeline.cypher_generator import generate_extraction_source_index_statements
        result = generate_extraction_source_index_statements(["TestCase", "Requirement", "Defect"])
        assert len(result) == 3

    def test_empty_labels_returns_empty(self):
        from graph_pipeline.cypher_generator import generate_extraction_source_index_statements
        assert generate_extraction_source_index_statements([]) == []

    def test_uses_create_index_not_constraint(self):
        from graph_pipeline.cypher_generator import generate_extraction_source_index_statements
        for stmt in generate_extraction_source_index_statements(["TestCase"]):
            assert "CREATE INDEX" in stmt
            assert "CONSTRAINT" not in stmt

    def test_uses_if_not_exists(self):
        from graph_pipeline.cypher_generator import generate_extraction_source_index_statements
        for stmt in generate_extraction_source_index_statements(["TestCase"]):
            assert "IF NOT EXISTS" in stmt

    def test_extraction_source_property_in_statement(self):
        from graph_pipeline.cypher_generator import generate_extraction_source_index_statements
        for stmt in generate_extraction_source_index_statements(["TestCase"]):
            assert "extraction_source" in stmt

    def test_label_appears_in_statement(self):
        from graph_pipeline.cypher_generator import generate_extraction_source_index_statements
        stmts = generate_extraction_source_index_statements(["Requirement"])
        assert any("Requirement" in s for s in stmts)


# ---------------------------------------------------------------------------
# generate_node_merge_batch / generate_relationship_merge_batch
# ---------------------------------------------------------------------------

def test_generate_node_merge_batch_contains_unwind():
    from graph_pipeline.cypher_generator import generate_node_merge_batch
    cypher = generate_node_merge_batch("TestCase")
    assert "UNWIND $rows AS row" in cypher
    assert "MERGE (n:TestCase {id: row.id})" in cypher
    assert "SET n += row.props" in cypher
    assert "SET n.ingested_at = datetime()" in cypher
    assert "row.extraction_source" in cypher


def test_generate_relationship_merge_batch_contains_unwind():
    from graph_pipeline.cypher_generator import generate_relationship_merge_batch
    cypher = generate_relationship_merge_batch("TestCase", "Requirement", "COVERS")
    assert "UNWIND $rows AS row" in cypher
    assert "MATCH (a:TestCase {id: row.from_id})" in cypher
    assert "MATCH (b:Requirement {id: row.to_id})" in cypher
    assert "MERGE (a)-[r:COVERS]->(b)" in cypher
    assert "row.extraction_source" in cypher


def test_generate_relationship_merge_batch_no_ingested_at():
    from graph_pipeline.cypher_generator import generate_relationship_merge_batch
    cypher = generate_relationship_merge_batch("A", "B", "REL")
    assert "ingested_at" not in cypher


def test_generate_relationship_merge_batch_empty_to_label_uses_labelless_match():
    from graph_pipeline.cypher_generator import generate_relationship_merge_batch
    cypher = generate_relationship_merge_batch("TestCase", "", "USES_MODULE")
    assert "MATCH (a:TestCase {id: row.from_id})" in cypher
    assert "MATCH (b {id: row.to_id})" in cypher
    assert "b: {" not in cypher  # no malformed "b: {" fragment


def test_generate_relationship_merge_batch_empty_from_label_uses_labelless_match():
    from graph_pipeline.cypher_generator import generate_relationship_merge_batch
    cypher = generate_relationship_merge_batch("", "XModule", "USES_MODULE")
    assert "MATCH (a {id: row.from_id})" in cypher
    assert "MATCH (b:XModule {id: row.to_id})" in cypher


def test_generate_node_merge_batch_label_is_interpolated():
    from graph_pipeline.cypher_generator import generate_node_merge_batch
    assert "XModule" in generate_node_merge_batch("XModule")
    assert "Folder" in generate_node_merge_batch("Folder")


# ---------------------------------------------------------------------------
# _validate_identifier — Cypher injection prevention
# ---------------------------------------------------------------------------

import pytest

class TestValidateIdentifier:
    """Covers every call-site that interpolates into Cypher strings."""

    # --- valid identifiers should pass without error ---

    def test_valid_label_passes_node_merge(self):
        from graph_pipeline.cypher_generator import generate_node_merge
        generate_node_merge(make_node(label="TestCase"))

    def test_valid_label_with_digits_passes(self):
        from graph_pipeline.cypher_generator import generate_node_merge_batch
        generate_node_merge_batch("XModule2")

    def test_valid_label_with_underscore_passes(self):
        from graph_pipeline.cypher_generator import generate_node_merge_batch
        generate_node_merge_batch("Test_Case")

    # --- invalid labels must raise ValueError ---

    def test_label_with_space_raises(self):
        from graph_pipeline.cypher_generator import generate_node_merge_batch
        with pytest.raises(ValueError, match="node label"):
            generate_node_merge_batch("Test Case")

    def test_label_with_hyphen_raises(self):
        from graph_pipeline.cypher_generator import generate_node_merge_batch
        with pytest.raises(ValueError, match="node label"):
            generate_node_merge_batch("Test-Case")

    def test_label_starting_with_digit_raises(self):
        from graph_pipeline.cypher_generator import generate_node_merge_batch
        with pytest.raises(ValueError, match="node label"):
            generate_node_merge_batch("1TestCase")

    def test_label_with_injection_payload_raises(self):
        from graph_pipeline.cypher_generator import generate_node_merge_batch
        with pytest.raises(ValueError, match="node label"):
            generate_node_merge_batch("TestCase} SET n.pwned=1 //")

    def test_empty_node_label_raises(self):
        from graph_pipeline.cypher_generator import generate_node_merge_batch
        with pytest.raises(ValueError, match="node label"):
            generate_node_merge_batch("")

    def test_rel_type_with_space_raises(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge_batch
        with pytest.raises(ValueError, match="relationship type"):
            generate_relationship_merge_batch("TestCase", "Requirement", "HAS CHILD")

    def test_rel_type_empty_raises(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge_batch
        with pytest.raises(ValueError, match="relationship type"):
            generate_relationship_merge_batch("TestCase", "Requirement", "")

    def test_rel_from_label_with_injection_raises(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge_batch
        with pytest.raises(ValueError, match="from_label"):
            generate_relationship_merge_batch("Bad Label!", "Requirement", "COVERS")

    def test_rel_to_label_with_injection_raises(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge_batch
        with pytest.raises(ValueError, match="to_label"):
            generate_relationship_merge_batch("TestCase", "Bad Label!", "COVERS")

    def test_constraint_invalid_label_raises(self):
        from graph_pipeline.cypher_generator import generate_constraint_statements
        with pytest.raises(ValueError, match="node label"):
            generate_constraint_statements(["ValidLabel", "Invalid Label"])

    def test_index_invalid_label_raises(self):
        from graph_pipeline.cypher_generator import generate_extraction_source_index_statements
        with pytest.raises(ValueError, match="node label"):
            generate_extraction_source_index_statements(["Bad-Label"])

    # --- empty optional labels (from_label / to_label) must still pass ---

    def test_empty_from_label_allowed_in_batch(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge_batch
        cypher = generate_relationship_merge_batch("", "Requirement", "COVERS")
        assert "MATCH (a {id: row.from_id})" in cypher

    def test_empty_to_label_allowed_in_batch(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge_batch
        cypher = generate_relationship_merge_batch("TestCase", "", "COVERS")
        assert "MATCH (b {id: row.to_id})" in cypher

    def test_empty_from_label_allowed_in_single_merge(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        rel = make_rel(from_label="", to_label="Requirement", type="COVERS")
        cypher, _ = generate_relationship_merge(rel)
        assert "MATCH (a {id: $from_id})" in cypher

    def test_empty_to_label_allowed_in_single_merge(self):
        from graph_pipeline.cypher_generator import generate_relationship_merge
        rel = make_rel(from_label="TestCase", to_label="", type="COVERS")
        cypher, _ = generate_relationship_merge(rel)
        assert "MATCH (b {id: $to_id})" in cypher
