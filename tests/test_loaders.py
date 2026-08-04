"""Tests for graph_pipeline models and loaders.

Run with: pytest tests/test_loaders.py
"""
import json as _json
import os
import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def fix(name):
    return os.path.join(FIXTURES, name)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class TestModels:
    def test_extraction_source_values(self):
        from graph_pipeline.models import ExtractionSource
        assert ExtractionSource.RULE_BASED == "rule_based"
        assert ExtractionSource.LLM_INFERRED == "llm_inferred"
        assert ExtractionSource.PHANTOM == "phantom"

    def test_node_fields(self):
        from graph_pipeline.models import Node, ExtractionSource
        n = Node(
            id="meap:abc123",
            label="TestCase",
            properties={"name": "Login Test"},
            source_record_id="abc123",
            extraction_source=ExtractionSource.RULE_BASED,
        )
        assert n.id == "meap:abc123"
        assert n.label == "TestCase"
        assert n.properties == {"name": "Login Test"}
        assert n.source_record_id == "abc123"
        assert n.extraction_source == ExtractionSource.RULE_BASED

    def test_relationship_fields(self):
        from graph_pipeline.models import Relationship, ExtractionSource
        r = Relationship(
            from_id="meap:abc123",
            to_id="meap:def456",
            from_label="TestCase",
            to_label="Folder",
            type="CONTAINS",
            properties={},
            source_record_id="abc123",
            extraction_source=ExtractionSource.PHANTOM,
        )
        assert r.from_id == "meap:abc123"
        assert r.to_id == "meap:def456"
        assert r.from_label == "TestCase"
        assert r.to_label == "Folder"
        assert r.type == "CONTAINS"
        assert r.extraction_source == ExtractionSource.PHANTOM

    def test_extraction_source_is_str_enum(self):
        from graph_pipeline.models import ExtractionSource
        # must be usable as a plain string (e.g. in Cypher SET n.extraction_source = $v)
        assert isinstance(ExtractionSource.RULE_BASED, str)


# ---------------------------------------------------------------------------
# JsonlLoader
# ---------------------------------------------------------------------------

class TestJsonlLoader:
    def test_normal_load(self):
        from graph_pipeline.loaders.jsonl_loader import JsonlLoader
        records = JsonlLoader().load(fix("sample.jsonl"))
        # header record is stripped; 3 data records remain
        assert len(records) == 3
        assert all(isinstance(r, dict) for r in records)
        assert records[0]["uniqueId"] == "abc123"

    def test_header_record_stripped(self):
        from graph_pipeline.loaders.jsonl_loader import JsonlLoader
        records = JsonlLoader().load(fix("sample.jsonl"))
        assert all(r.get("kind") != "export-dump-header" for r in records)

    def test_empty_file(self):
        from graph_pipeline.loaders.jsonl_loader import JsonlLoader
        records = JsonlLoader().load(fix("empty.jsonl"))
        assert records == []

    def test_header_only_file(self):
        from graph_pipeline.loaders.jsonl_loader import JsonlLoader
        records = JsonlLoader().load(fix("header_only.jsonl"))
        assert records == []

    def test_blank_lines_skipped(self):
        from graph_pipeline.loaders.jsonl_loader import JsonlLoader
        # header_only.jsonl has a trailing blank line
        records = JsonlLoader().load(fix("header_only.jsonl"))
        assert records == []

    def test_can_handle_jsonl(self):
        from graph_pipeline.loaders.jsonl_loader import JsonlLoader
        loader = JsonlLoader()
        assert loader.can_handle("data.jsonl")
        assert loader.can_handle("data.ndjson")
        assert not loader.can_handle("data.json")
        assert not loader.can_handle("data.csv")

    def test_type_distribution_preserved(self):
        from graph_pipeline.loaders.jsonl_loader import JsonlLoader
        records = JsonlLoader().load(fix("sample.jsonl"))
        types = [r["typeName"] for r in records]
        assert types.count("TestCase") == 2
        assert types.count("XModule") == 1


# ---------------------------------------------------------------------------
# JsonLoader
# ---------------------------------------------------------------------------

class TestJsonLoader:
    def test_root_array(self):
        from graph_pipeline.loaders.json_loader import JsonLoader
        records = JsonLoader().load(fix("sample_array.json"))
        assert len(records) == 2
        assert records[0]["name"] == "Alice"

    def test_root_object_with_single_array(self):
        from graph_pipeline.loaders.json_loader import JsonLoader
        records = JsonLoader().load(fix("sample_object_with_array.json"))
        assert len(records) == 2
        assert records[0]["label"] == "foo"

    def test_nested_object_flattened(self):
        from graph_pipeline.loaders.json_loader import JsonLoader
        records = JsonLoader().load(fix("sample_nested_object.json"))
        assert len(records) == 2
        keys = {r.get("_key") for r in records}
        assert keys == {"alpha", "beta"}
        values = {r.get("x") for r in records}
        assert values == {1, 3}

    def test_empty_array(self):
        from graph_pipeline.loaders.json_loader import JsonLoader
        records = JsonLoader().load(fix("empty.json"))
        assert records == []

    def test_can_handle_json(self):
        from graph_pipeline.loaders.json_loader import JsonLoader
        loader = JsonLoader()
        assert loader.can_handle("data.json")
        assert not loader.can_handle("data.jsonl")
        assert not loader.can_handle("data.csv")


# ---------------------------------------------------------------------------
# CsvLoader
# ---------------------------------------------------------------------------

class TestCsvLoader:
    def test_normal_load(self):
        from graph_pipeline.loaders.csv_loader import CsvLoader
        records = CsvLoader().load(fix("sample.csv"))
        assert len(records) == 3
        assert records[0]["name"] == "Alice"

    def test_numeric_coercion(self):
        from graph_pipeline.loaders.csv_loader import CsvLoader
        records = CsvLoader().load(fix("sample.csv"))
        assert records[0]["age"] == 30
        assert isinstance(records[0]["age"], int)
        assert records[0]["score"] == 9.5
        assert isinstance(records[0]["score"], float)

    def test_empty_cells_are_none(self):
        from graph_pipeline.loaders.csv_loader import CsvLoader
        records = CsvLoader().load(fix("sample.csv"))
        # Alice has no notes; Charlie has no age
        assert records[0]["notes"] is None
        assert records[2]["age"] is None

    def test_empty_file(self):
        from graph_pipeline.loaders.csv_loader import CsvLoader
        records = CsvLoader().load(fix("empty.csv"))
        assert records == []

    def test_tsv(self):
        from graph_pipeline.loaders.csv_loader import CsvLoader
        records = CsvLoader().load(fix("sample.tsv"))
        assert len(records) == 2
        assert records[0]["city"] == "Berlin"
        assert records[0]["population"] == 3645000

    def test_can_handle(self):
        from graph_pipeline.loaders.csv_loader import CsvLoader
        loader = CsvLoader()
        assert loader.can_handle("data.csv")
        assert loader.can_handle("data.tsv")
        assert not loader.can_handle("data.json")


# ---------------------------------------------------------------------------
# SqlLoader
# ---------------------------------------------------------------------------

class TestSqlLoader:
    def test_normal_load(self):
        from graph_pipeline.loaders.sql_loader import SqlLoader
        records = SqlLoader().load(fix("sample.sqlite"))
        assert len(records) == 4  # 2 users + 2 orders
        tables = {r["_table"] for r in records}
        assert tables == {"users", "orders"}

    def test_table_property_set(self):
        from graph_pipeline.loaders.sql_loader import SqlLoader
        records = SqlLoader().load(fix("sample.sqlite"))
        for r in records:
            assert "_table" in r

    def test_foreign_key_preserved(self):
        from graph_pipeline.loaders.sql_loader import SqlLoader
        records = SqlLoader().load(fix("sample.sqlite"))
        orders = [r for r in records if r["_table"] == "orders"]
        assert all("user_id" in r for r in orders)

    def test_empty_table(self, tmp_path):
        import sqlite3
        from graph_pipeline.loaders.sql_loader import SqlLoader
        db = str(tmp_path / "empty.sqlite")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE empty_tbl (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        records = SqlLoader().load(db)
        assert records == []

    def test_can_handle(self):
        from graph_pipeline.loaders.sql_loader import SqlLoader
        loader = SqlLoader()
        assert loader.can_handle("data.sqlite")
        assert loader.can_handle("data.db")
        assert not loader.can_handle("data.sql")
        assert not loader.can_handle("data.csv")


# ---------------------------------------------------------------------------
# Auto-detection (__init__.py load())
# ---------------------------------------------------------------------------

class TestAutoDetect:
    def test_detects_jsonl(self):
        from graph_pipeline.loaders import load
        records = load(fix("sample.jsonl"))
        assert len(records) == 3

    def test_detects_json(self):
        from graph_pipeline.loaders import load
        records = load(fix("sample_array.json"))
        assert len(records) == 2

    def test_detects_csv(self):
        from graph_pipeline.loaders import load
        records = load(fix("sample.csv"))
        assert len(records) == 3

    def test_detects_sqlite(self):
        from graph_pipeline.loaders import load
        records = load(fix("sample.sqlite"))
        assert len(records) == 4

    def test_unknown_extension_raises(self):
        from graph_pipeline.loaders import load
        with pytest.raises(ValueError, match="No loader found"):
            load("data.parquet")

    def test_detects_ndjson(self):
        from graph_pipeline.loaders import load
        records = load(fix("sample.jsonl"))
        assert isinstance(records, list)


# ---------------------------------------------------------------------------
# Helpers for stream tests
# ---------------------------------------------------------------------------

def _record_set(records):
    """Convert records to a set of JSON strings for order-independent comparison."""
    return {_json.dumps(r, sort_keys=True, default=str) for r in records}


# ---------------------------------------------------------------------------
# JsonlLoader stream
# ---------------------------------------------------------------------------

class TestJsonlLoaderStream:
    def test_stream_jsonl_yields_same_records_as_load(self):
        from graph_pipeline.loaders.jsonl_loader import JsonlLoader
        loader = JsonlLoader()
        loaded = loader.load(fix("sample.jsonl"))
        streamed = list(loader.stream(fix("sample.jsonl")))
        assert _record_set(loaded) == _record_set(streamed)

    def test_stream_jsonl_skips_header_record(self):
        from graph_pipeline.loaders.jsonl_loader import JsonlLoader
        streamed = list(JsonlLoader().stream(fix("sample.jsonl")))
        assert all(r.get("kind") != "export-dump-header" for r in streamed)


# ---------------------------------------------------------------------------
# CsvLoader stream
# ---------------------------------------------------------------------------

class TestCsvLoaderStream:
    def test_stream_csv_yields_same_records_as_load(self):
        from graph_pipeline.loaders.csv_loader import CsvLoader
        loader = CsvLoader()
        loaded = loader.load(fix("sample.csv"))
        streamed = list(loader.stream(fix("sample.csv")))
        assert _record_set(loaded) == _record_set(streamed)

    def test_stream_tsv_yields_same_records_as_load(self):
        from graph_pipeline.loaders.csv_loader import CsvLoader
        loader = CsvLoader()
        loaded = loader.load(fix("sample.tsv"))
        streamed = list(loader.stream(fix("sample.tsv")))
        assert _record_set(loaded) == _record_set(streamed)


# ---------------------------------------------------------------------------
# SqlLoader stream
# ---------------------------------------------------------------------------

class TestSqlLoaderStream:
    def test_stream_sql_yields_same_records_as_load(self):
        from graph_pipeline.loaders.sql_loader import SqlLoader
        loader = SqlLoader()
        loaded = loader.load(fix("sample.sqlite"))
        streamed = list(loader.stream(fix("sample.sqlite")))
        assert _record_set(loaded) == _record_set(streamed)


# ---------------------------------------------------------------------------
# JsonLoader stream
# ---------------------------------------------------------------------------

class TestJsonLoaderStream:
    def test_stream_json_yields_same_records_as_load(self):
        from graph_pipeline.loaders.json_loader import JsonLoader
        loader = JsonLoader()
        loaded = loader.load(fix("sample_array.json"))
        streamed = list(loader.stream(fix("sample_array.json")))
        assert _record_set(loaded) == _record_set(streamed)


# ---------------------------------------------------------------------------
# Auto-detection stream()
# ---------------------------------------------------------------------------

class TestAutoDetectStream:
    def test_stream_top_level_function_dispatches_to_loader(self):
        from graph_pipeline.loaders import load, stream
        loaded = load(fix("sample.jsonl"))
        streamed = list(stream(fix("sample.jsonl")))
        assert _record_set(loaded) == _record_set(streamed)
