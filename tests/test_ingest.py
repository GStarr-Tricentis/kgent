"""Tests for the _schema_preview helper and connectivity check in scripts/ingest.py."""
import importlib.util
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _load_ingest_module():
    spec = importlib.util.spec_from_file_location(
        "scripts_ingest",
        Path(__file__).parent.parent / "scripts" / "ingest.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ingest_mod = _load_ingest_module()
_schema_preview = _ingest_mod._schema_preview
_ingest_main = _ingest_mod.main


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


# ---------------------------------------------------------------------------
# Neo4j connectivity check
# ---------------------------------------------------------------------------

def _make_mock_config(tmp_path):
    cfg = MagicMock()
    cfg.graph_pipeline.default_sample_size = 10
    cfg.graph_pipeline.default_batch_size = 100
    cfg.graph_pipeline.context_dir = str(tmp_path)
    return cfg


def _make_mock_prescan_sample():
    """PrescanSampleResult-shaped mock."""
    m = MagicMock()
    m.sample = [{"uniqueId": "tc-001"}]
    m.fingerprint = "fp123"
    m.total_records = 1
    m.type_counts = Counter({"TestCase": 1})
    return m


def _make_mock_hash_diff():
    """HashDiffResult-shaped mock with one record to ingest (avoids the early-exit path)."""
    m = MagicMock()
    m.ingest_ids = {"tc-001"}
    m.deleted_ids = set()
    m.current_hashes = {"tc-001": "h1"}
    return m


def _make_prior_ctx():
    """DatasetContext mock with matching fingerprint so schema discovery and review are skipped."""
    ctx = MagicMock()
    ctx.source_fingerprint = "fp123"
    ctx.id_field = "uniqueId"
    ctx.node_types = []
    ctx.relationship_types = []
    ctx.ambiguous_fields = []
    ctx.ambiguous_field_rules = []
    ctx.nested_collections = []
    ctx.hierarchy_config = None
    return ctx


class TestIngestConnectivityCheck:
    """Verify that verify_connectivity is called early and failures are handled cleanly."""

    def _apply_base_patches(self, monkeypatch, tmp_path):
        """Patch all setup steps so main() reaches the driver-construction block."""
        monkeypatch.setattr(
            "sys.argv",
            ["ingest.py", "--file", str(tmp_path / "dummy.jsonl"), "--skip-review"],
        )
        monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
        monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
        monkeypatch.setenv("NEO4J_PASSWORD", "testpassword")

    async def test_verify_connectivity_failure_exits_1(self, tmp_path, monkeypatch):
        """A connectivity failure causes sys.exit(1) before any extraction runs."""
        self._apply_base_patches(monkeypatch, tmp_path)

        mock_driver = AsyncMock()
        mock_driver.verify_connectivity.side_effect = Exception("connection refused")

        with patch("kgent.config.loader.load_dotenv"), \
             patch("kgent.config.loader.load_config", return_value=_make_mock_config(tmp_path)), \
             patch("kgent.models.factory.make_backend", new_callable=AsyncMock, return_value=MagicMock()), \
             patch("graph_pipeline.loaders.stream", return_value=iter([])), \
             patch("graph_pipeline.sampler.prescan_sample", return_value=_make_mock_prescan_sample()), \
             patch("graph_pipeline.sampler.compute_hash_diff", return_value=_make_mock_hash_diff()), \
             patch("graph_pipeline.context_store.load_record_hashes", return_value={}), \
             patch("graph_pipeline.context_store.load_shared_context", return_value=MagicMock(version=1, node_types=[])), \
             patch("graph_pipeline.context_store.load_dataset_context", return_value=_make_prior_ctx()), \
             patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver):
            with pytest.raises(SystemExit) as exc_info:
                await _ingest_main()

        assert exc_info.value.code == 1
        mock_driver.verify_connectivity.assert_awaited_once()

    async def test_verify_connectivity_failure_closes_driver(self, tmp_path, monkeypatch):
        """Driver is closed cleanly when verify_connectivity fails."""
        self._apply_base_patches(monkeypatch, tmp_path)

        mock_driver = AsyncMock()
        mock_driver.verify_connectivity.side_effect = Exception("auth error")

        with patch("kgent.config.loader.load_dotenv"), \
             patch("kgent.config.loader.load_config", return_value=_make_mock_config(tmp_path)), \
             patch("kgent.models.factory.make_backend", new_callable=AsyncMock, return_value=MagicMock()), \
             patch("graph_pipeline.loaders.stream", return_value=iter([])), \
             patch("graph_pipeline.sampler.prescan_sample", return_value=_make_mock_prescan_sample()), \
             patch("graph_pipeline.sampler.compute_hash_diff", return_value=_make_mock_hash_diff()), \
             patch("graph_pipeline.context_store.load_record_hashes", return_value={}), \
             patch("graph_pipeline.context_store.load_shared_context", return_value=MagicMock(version=1, node_types=[])), \
             patch("graph_pipeline.context_store.load_dataset_context", return_value=_make_prior_ctx()), \
             patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver):
            with pytest.raises(SystemExit):
                await _ingest_main()

        mock_driver.close.assert_awaited_once()

    async def test_verify_connectivity_called_before_extraction(self, tmp_path, monkeypatch):
        """verify_connectivity is called before build_extraction_indices."""
        self._apply_base_patches(monkeypatch, tmp_path)

        mock_driver = AsyncMock()
        mock_driver.verify_connectivity.side_effect = Exception("connection refused")
        mock_build_indices = MagicMock()

        with patch("kgent.config.loader.load_dotenv"), \
             patch("kgent.config.loader.load_config", return_value=_make_mock_config(tmp_path)), \
             patch("kgent.models.factory.make_backend", new_callable=AsyncMock, return_value=MagicMock()), \
             patch("graph_pipeline.loaders.stream", return_value=iter([])), \
             patch("graph_pipeline.sampler.prescan_sample", return_value=_make_mock_prescan_sample()), \
             patch("graph_pipeline.sampler.compute_hash_diff", return_value=_make_mock_hash_diff()), \
             patch("graph_pipeline.context_store.load_record_hashes", return_value={}), \
             patch("graph_pipeline.context_store.load_shared_context", return_value=MagicMock(version=1, node_types=[])), \
             patch("graph_pipeline.context_store.load_dataset_context", return_value=_make_prior_ctx()), \
             patch("graph_pipeline.extractor.build_extraction_indices", mock_build_indices), \
             patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver):
            with pytest.raises(SystemExit):
                await _ingest_main()

        mock_driver.verify_connectivity.assert_awaited_once()
        mock_build_indices.assert_not_called()

    async def test_dry_run_skips_driver_construction(self, tmp_path, monkeypatch):
        """--dry-run never constructs an AsyncDriver or calls verify_connectivity."""
        monkeypatch.setattr(
            "sys.argv",
            ["ingest.py", "--file", str(tmp_path / "dummy.jsonl"),
             "--skip-review", "--dry-run"],
        )

        mock_neo4j_driver = MagicMock()
        prior_ctx = _make_prior_ctx()

        with patch("kgent.config.loader.load_dotenv"), \
             patch("kgent.config.loader.load_config", return_value=_make_mock_config(tmp_path)), \
             patch("kgent.models.factory.make_backend", new_callable=AsyncMock, return_value=MagicMock()), \
             patch("graph_pipeline.loaders.stream", return_value=iter([])), \
             patch("graph_pipeline.sampler.prescan_sample", return_value=_make_mock_prescan_sample()), \
             patch("graph_pipeline.sampler.compute_hash_diff", return_value=_make_mock_hash_diff()), \
             patch("graph_pipeline.context_store.load_record_hashes", return_value={}), \
             patch("graph_pipeline.context_store.load_shared_context", return_value=MagicMock(version=1, node_types=[])), \
             patch("graph_pipeline.context_store.load_dataset_context", return_value=prior_ctx), \
             patch("graph_pipeline.extractor.build_extraction_indices", return_value=MagicMock()), \
             patch("graph_pipeline.validator.check_label_coverage", return_value=[]), \
             patch("neo4j.AsyncGraphDatabase.driver", mock_neo4j_driver):
            await _ingest_main()

        mock_neo4j_driver.assert_not_called()


# ---------------------------------------------------------------------------
# id_field routing regression
# ---------------------------------------------------------------------------

class TestIngestIdFieldRouting:
    async def test_compute_hash_diff_called_with_dataset_ctx_id_field(self, tmp_path, monkeypatch):
        """compute_hash_diff receives dataset_ctx.id_field, never the hardcoded 'uniqueId'."""
        monkeypatch.setattr(
            "sys.argv",
            ["ingest.py", "--file", str(tmp_path / "dummy.jsonl"), "--skip-review"],
        )
        monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
        monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
        monkeypatch.setenv("NEO4J_PASSWORD", "testpassword")

        prior_ctx = _make_prior_ctx()
        prior_ctx.id_field = "customId"  # non-default id_field

        mock_driver = AsyncMock()
        mock_driver.verify_connectivity.side_effect = Exception("stop here")
        mock_hash_diff = MagicMock()

        with patch("kgent.config.loader.load_dotenv"), \
             patch("kgent.config.loader.load_config", return_value=_make_mock_config(tmp_path)), \
             patch("kgent.models.factory.make_backend", new_callable=AsyncMock, return_value=MagicMock()), \
             patch("graph_pipeline.loaders.stream", return_value=iter([])), \
             patch("graph_pipeline.sampler.prescan_sample", return_value=_make_mock_prescan_sample()), \
             patch("graph_pipeline.sampler.compute_hash_diff", mock_hash_diff) as patched_diff, \
             patch("graph_pipeline.context_store.load_record_hashes", return_value={}), \
             patch("graph_pipeline.context_store.load_shared_context", return_value=MagicMock(version=1, node_types=[])), \
             patch("graph_pipeline.context_store.load_dataset_context", return_value=prior_ctx), \
             patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver):
            with pytest.raises(SystemExit):
                await _ingest_main()

        # compute_hash_diff must have been called with id_field="customId", not "uniqueId"
        assert patched_diff.called
        _, kwargs = patched_diff.call_args
        assert kwargs.get("id_field") == "customId" or patched_diff.call_args.args[1] == "customId"
