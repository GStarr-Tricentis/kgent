"""Tests for the _schema_preview helper in scripts/ingest.py."""
import importlib.util
from pathlib import Path

import pytest


def _load_schema_preview():
    spec = importlib.util.spec_from_file_location(
        "scripts_ingest",
        Path(__file__).parent.parent / "scripts" / "ingest.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._schema_preview


_schema_preview = _load_schema_preview()


def _make_ctx(**overrides):
    from graph_pipeline.context_store import (
        DatasetContext,
        DatasetNodeType,
        DatasetRelationshipType,
    )
    defaults = dict(
        dataset_id="test_ds",
        id_field="uniqueId",
        type_field="typeName",
        node_types=[
            DatasetNodeType(name="TestCase", maps_to="TestCase"),
            DatasetNodeType(name="Req", maps_to="Requirement"),
        ],
        relationship_types=[
            DatasetRelationshipType(
                name="COVERS", maps_to="COVERS",
                **{"from": "TestCase", "to": "Requirement"},
            ),
        ],
    )
    defaults.update(overrides)
    return DatasetContext(**defaults)


class TestSchemaPreview:
    def test_contains_header(self):
        out = _schema_preview(_make_ctx())
        assert "dry-run" in out.lower()
        assert "not saved" in out.lower()

    def test_id_and_type_field_shown(self):
        out = _schema_preview(_make_ctx(id_field="myId", type_field="myType"))
        assert "myId" in out
        assert "myType" in out

    def test_node_types_listed(self):
        out = _schema_preview(_make_ctx())
        assert "TestCase" in out
        assert "Requirement" in out

    def test_relationship_types_listed(self):
        out = _schema_preview(_make_ctx())
        assert "COVERS" in out
        assert "TestCase" in out
        assert "Requirement" in out

    def test_empty_optional_sections_omitted(self):
        ctx = _make_ctx(
            node_types=[],
            relationship_types=[],
            ambiguous_fields=[],
            implicit_relationships=[],
        )
        out = _schema_preview(ctx)
        assert "node_types" not in out
        assert "relationship_types" not in out
        assert "ambiguous_fields" not in out

    def test_ambiguous_fields_listed(self):
        from graph_pipeline.context_store import DatasetContext, DatasetNodeType
        ctx = DatasetContext(
            dataset_id="ds",
            node_types=[DatasetNodeType(name="TC", maps_to="TC")],
            ambiguous_fields=["notes", "description"],
        )
        out = _schema_preview(ctx)
        assert "notes" in out
        assert "description" in out
