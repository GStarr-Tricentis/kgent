"""Tests for graph_pipeline/context_store.py.

Run with: pytest tests/test_context_store.py
"""
import os
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_dataset_ctx(
    dataset_id="ds1",
    source_file="ds1.jsonl",
    node_types=None,
    relationship_types=None,
    implicit_relationships=None,
    design_decisions=None,
):
    from graph_pipeline.context_store import (
        DatasetContext,
        DatasetNodeType,
        DatasetRelationshipType,
    )

    return DatasetContext(
        dataset_id=dataset_id,
        source_file=source_file,
        node_types=node_types or [],
        relationship_types=relationship_types or [],
        implicit_relationships=implicit_relationships or [],
        design_decisions=design_decisions or [],
    )


def make_node_type(name, maps_to, identity_key="uniqueId"):
    from graph_pipeline.context_store import DatasetNodeType
    return DatasetNodeType(name=name, maps_to=maps_to, identity_key=identity_key)


def make_rel_type(name, maps_to, from_type, to_type):
    from graph_pipeline.context_store import DatasetRelationshipType
    return DatasetRelationshipType(
        name=name, maps_to=maps_to, from_type=from_type, to_type=to_type
    )


# ---------------------------------------------------------------------------
# SharedContext / DatasetContext models
# ---------------------------------------------------------------------------

class TestModels:
    def test_shared_context_empty_defaults(self):
        from graph_pipeline.context_store import SharedContext
        sc = SharedContext()
        assert sc.version == 0
        assert sc.node_types == []
        assert sc.relationship_types == []
        assert sc.structural_patterns == []

    def test_dataset_context_fields(self):
        ctx = make_dataset_ctx(
            dataset_id="meap",
            source_file="dump.jsonl",
            node_types=[make_node_type("TestCase", "TestCase")],
        )
        assert ctx.dataset_id == "meap"
        assert ctx.source_file == "dump.jsonl"
        assert len(ctx.node_types) == 1
        assert ctx.node_types[0].name == "TestCase"
        assert ctx.node_types[0].maps_to == "TestCase"

    def test_implicit_relationship_fields(self):
        from graph_pipeline.context_store import ImplicitRelationship
        ir = ImplicitRelationship(
            description="associations[].edgeName='Module' → USES_MODULE",
            pattern="associations_edge",
            edge_name="Module",
            maps_to="USES_MODULE",
            cross_dataset=False,
            target_dataset_id=None,
        )
        assert ir.cross_dataset is False
        assert ir.target_dataset_id is None

    def test_design_decision_fields(self):
        from graph_pipeline.context_store import DesignDecision
        dd = DesignDecision(
            question="Nodes or properties?",
            decision="nodes",
            rationale="They have uniqueIds",
        )
        assert dd.decision == "nodes"


# ---------------------------------------------------------------------------
# load_shared_context — missing file
# ---------------------------------------------------------------------------

class TestLoadSharedContext:
    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        from graph_pipeline import context_store
        # reload so env var is picked up
        import importlib; importlib.reload(context_store)
        sc = context_store.load_shared_context()
        assert sc.version == 0
        assert sc.node_types == []
        assert sc.relationship_types == []

    def test_existing_file_loads(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        yaml_content = """\
version: 2
updated_at: "2026-01-01"
node_types:
  - name: TestCase
    description: "A test procedure"
    identity_key: uniqueId
    source_datasets: [meap]
relationship_types: []
structural_patterns: []
"""
        (tmp_path / "shared_context.yaml").write_text(yaml_content)
        sc = context_store.load_shared_context()
        assert sc.version == 2
        assert len(sc.node_types) == 1
        assert sc.node_types[0].name == "TestCase"
        assert sc.node_types[0].source_datasets == ["meap"]

    def test_malformed_shared_context_raises_value_error(self, tmp_path, monkeypatch):
        """A corrupted shared_context.yaml raises ValueError containing the file path."""
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        # node_types must be a list; a string value triggers validation failure in _shared_from_dict
        bad_yaml = "version: 1\nnode_types: not_a_list\nrelationship_types: []\n"
        (tmp_path / "shared_context.yaml").write_text(bad_yaml)

        with pytest.raises(ValueError) as exc_info:
            context_store.load_shared_context()

        assert "shared_context.yaml" in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_dataset_context / save_dataset_context
# ---------------------------------------------------------------------------

class TestDatasetContextIO:
    def test_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)
        assert context_store.load_dataset_context("nonexistent") is None

    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        ctx = make_dataset_ctx(
            dataset_id="meap",
            source_file="dump.jsonl",
            node_types=[make_node_type("TestCase", "TestCase")],
        )
        context_store.save_dataset_context(ctx)
        loaded = context_store.load_dataset_context("meap")
        assert loaded is not None
        assert loaded.dataset_id == "meap"
        assert loaded.node_types[0].maps_to == "TestCase"

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        # datasets/ sub-directory does not exist yet
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        ctx = make_dataset_ctx(dataset_id="new_ds")
        context_store.save_dataset_context(ctx)
        assert (tmp_path / "datasets" / "new_ds.yaml").exists()

    def test_malformed_dataset_context_raises_value_error(self, tmp_path, monkeypatch):
        """A YAML file that fails DatasetContext validation raises ValueError, not a raw Pydantic error."""
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        # dataset_id is a required field; omitting it triggers validation failure
        bad_yaml = "source_file: something.jsonl\nnode_types: []\n"
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets" / "bad_ds.yaml").write_text(bad_yaml)

        with pytest.raises(ValueError) as exc_info:
            context_store.load_dataset_context("bad_ds")

        assert "bad_ds.yaml" in str(exc_info.value)

    def test_malformed_dataset_context_error_contains_hint(self, tmp_path, monkeypatch):
        """The ValueError message tells the user how to recover."""
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        bad_yaml = "source_file: something.jsonl\nnode_types: []\n"
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets" / "bad_ds2.yaml").write_text(bad_yaml)

        with pytest.raises(ValueError) as exc_info:
            context_store.load_dataset_context("bad_ds2")

        msg = str(exc_info.value)
        assert "Fix the YAML file" in msg or "delete it" in msg

    def test_malformed_dataset_context_preserves_cause(self, tmp_path, monkeypatch):
        """The original exception is attached as __cause__ so full tracebacks still show it."""
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        bad_yaml = "source_file: something.jsonl\nnode_types: []\n"
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets" / "bad_ds3.yaml").write_text(bad_yaml)

        with pytest.raises(ValueError) as exc_info:
            context_store.load_dataset_context("bad_ds3")

        assert exc_info.value.__cause__ is not None

    def test_source_fingerprint_persists_through_save_load(self, tmp_path, monkeypatch):
        """source_fingerprint written to YAML is read back with the correct value."""
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        ctx = context_store.DatasetContext(
            dataset_id="fp_test",
            source_fingerprint="abc123def45678",
        )
        context_store.save_dataset_context(ctx)
        reloaded = context_store.load_dataset_context("fp_test")
        assert reloaded.source_fingerprint == "abc123def45678"

    def test_source_fingerprint_defaults_to_empty_on_old_yaml(self, tmp_path, monkeypatch):
        """Context files without source_fingerprint load with empty string default."""
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets" / "old_ds.yaml").write_text(
            "dataset_id: old_ds\nsource_file: ''\n"
        )
        loaded = context_store.load_dataset_context("old_ds")
        assert loaded.source_fingerprint == ""


# ---------------------------------------------------------------------------
# merge_into_shared — clean new type
# ---------------------------------------------------------------------------

class TestMergeNewType:
    def _reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)
        return context_store

    def test_new_node_type_appended(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[make_node_type("TestCase", "TestCase")],
        )
        merged = cs.merge_into_shared(ctx)
        assert len(merged.node_types) == 1
        assert merged.node_types[0].name == "TestCase"
        assert "ds1" in merged.node_types[0].source_datasets

    def test_version_incremented(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx = make_dataset_ctx(node_types=[make_node_type("TestCase", "TestCase")])
        merged = cs.merge_into_shared(ctx)
        assert merged.version == 1  # started at 0, incremented to 1

    def test_updated_at_set_to_today(self, tmp_path, monkeypatch):
        import datetime
        cs = self._reload(tmp_path, monkeypatch)
        ctx = make_dataset_ctx(node_types=[make_node_type("TestCase", "TestCase")])
        merged = cs.merge_into_shared(ctx)
        assert merged.updated_at == str(datetime.date.today())

    def test_new_relationship_type_appended(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            relationship_types=[make_rel_type("covers", "COVERS", "TestCase", "Requirement")],
        )
        merged = cs.merge_into_shared(ctx)
        assert len(merged.relationship_types) == 1
        assert merged.relationship_types[0].name == "covers"
        assert merged.relationship_types[0].maps_to == "COVERS"
        assert "ds1" in merged.relationship_types[0].source_datasets

    def test_merge_persists_to_disk(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx = make_dataset_ctx(node_types=[make_node_type("TestCase", "TestCase")])
        cs.merge_into_shared(ctx)
        # reload from disk to confirm persistence
        import importlib; importlib.reload(cs)
        sc = cs.load_shared_context()
        assert sc.version == 1


# ---------------------------------------------------------------------------
# merge_into_shared — same source name, same canonical (no-op / dedup)
# ---------------------------------------------------------------------------

class TestMergeSameType:
    def _reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)
        return context_store

    def test_same_canonical_no_duplicate_node(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx1 = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[make_node_type("TestCase", "TestCase")],
        )
        ctx2 = make_dataset_ctx(
            dataset_id="ds2",
            node_types=[make_node_type("TestCase", "TestCase")],
        )
        cs.merge_into_shared(ctx1)
        merged = cs.merge_into_shared(ctx2)
        assert len(merged.node_types) == 1  # still just one entry

    def test_same_canonical_source_dataset_appended(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx1 = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[make_node_type("TestCase", "TestCase")],
        )
        ctx2 = make_dataset_ctx(
            dataset_id="ds2",
            node_types=[make_node_type("TestCase", "TestCase")],
        )
        cs.merge_into_shared(ctx1)
        merged = cs.merge_into_shared(ctx2)
        assert set(merged.node_types[0].source_datasets) == {"ds1", "ds2"}

    def test_version_still_increments_on_dedup(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx1 = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[make_node_type("TestCase", "TestCase")],
        )
        ctx2 = make_dataset_ctx(
            dataset_id="ds2",
            node_types=[make_node_type("TestCase", "TestCase")],
        )
        cs.merge_into_shared(ctx1)
        merged = cs.merge_into_shared(ctx2)
        assert merged.version == 2

    def test_same_dataset_id_not_duplicated_in_source_datasets(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[make_node_type("TestCase", "TestCase")],
        )
        cs.merge_into_shared(ctx)
        merged = cs.merge_into_shared(ctx)  # same dataset, same type, re-ingested
        assert merged.node_types[0].source_datasets.count("ds1") == 1


# ---------------------------------------------------------------------------
# merge_into_shared — conflict: same source name, different canonical
# ---------------------------------------------------------------------------

class TestMergeConflict:
    def _reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)
        return context_store

    def test_conflict_raises(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx1 = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[make_node_type("ReuseableTestStepBlock", "ReusableStep")],
        )
        ctx2 = make_dataset_ctx(
            dataset_id="ds2",
            node_types=[make_node_type("ReuseableTestStepBlock", "ReuseableBlock")],
        )
        cs.merge_into_shared(ctx1)
        with pytest.raises(cs.MergeConflict):
            cs.merge_into_shared(ctx2)

    def test_conflict_fields_correct(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx1 = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[make_node_type("ReuseableTestStepBlock", "ReusableStep")],
        )
        ctx2 = make_dataset_ctx(
            dataset_id="ds2",
            node_types=[make_node_type("ReuseableTestStepBlock", "ReuseableBlock")],
        )
        cs.merge_into_shared(ctx1)
        with pytest.raises(cs.MergeConflict) as exc_info:
            cs.merge_into_shared(ctx2)
        conflict = exc_info.value
        assert conflict.source_name == "ReuseableTestStepBlock"
        assert conflict.existing_canonical == "ReusableStep"
        assert conflict.proposed_canonical == "ReuseableBlock"
        assert conflict.existing_dataset == "ds1"
        assert conflict.new_dataset == "ds2"
        assert conflict.type == "node"

    def test_relationship_conflict_raises(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx1 = make_dataset_ctx(
            dataset_id="ds1",
            relationship_types=[make_rel_type("covers", "COVERS", "TestCase", "Requirement")],
        )
        ctx2 = make_dataset_ctx(
            dataset_id="ds2",
            relationship_types=[make_rel_type("covers", "COVERS_REQ", "TestCase", "Requirement")],
        )
        cs.merge_into_shared(ctx1)
        with pytest.raises(cs.MergeConflict) as exc_info:
            cs.merge_into_shared(ctx2)
        conflict = exc_info.value
        assert conflict.type == "relationship"
        assert conflict.source_name == "covers"
        assert conflict.existing_canonical == "COVERS"
        assert conflict.proposed_canonical == "COVERS_REQ"

    def test_conflict_does_not_mutate_shared(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        ctx1 = make_dataset_ctx(
            dataset_id="ds1",
            node_types=[make_node_type("ReuseableTestStepBlock", "ReusableStep")],
        )
        ctx2 = make_dataset_ctx(
            dataset_id="ds2",
            node_types=[make_node_type("ReuseableTestStepBlock", "ReuseableBlock")],
        )
        cs.merge_into_shared(ctx1)
        with pytest.raises(cs.MergeConflict):
            cs.merge_into_shared(ctx2)
        # shared context on disk must be unchanged (version still 1, not 2)
        import importlib; importlib.reload(cs)
        sc = cs.load_shared_context()
        assert sc.version == 1
        assert sc.node_types[0].maps_to == "ReusableStep"


# ---------------------------------------------------------------------------
# merge_into_shared — file locking
# ---------------------------------------------------------------------------

class TestMergeSharedLocking:
    """Verify that merge_into_shared holds and releases an exclusive file lock."""

    def test_lock_file_created_after_merge(self, tmp_path, monkeypatch):
        """A .lock file sibling of shared_context.yaml is created during merge."""
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        ctx = context_store.DatasetContext(dataset_id="ds1")
        context_store.merge_into_shared(ctx)

        assert (tmp_path / "shared_context.lock").exists()

    def test_lock_released_after_merge(self, tmp_path, monkeypatch):
        """The exclusive lock is released when merge_into_shared returns normally."""
        import fcntl
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)

        ctx = context_store.DatasetContext(dataset_id="ds1")
        context_store.merge_into_shared(ctx)

        # LOCK_EX | LOCK_NB raises BlockingIOError if the lock is still held;
        # if it succeeds the lock was cleanly released.
        lock_path = tmp_path / "shared_context.lock"
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lf, fcntl.LOCK_UN)

    def test_concurrent_merges_do_not_lose_data(self, tmp_path, monkeypatch):
        """Five threads merging distinct datasets all survive; no type is silently dropped."""
        import importlib
        import threading
        from graph_pipeline import context_store
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        importlib.reload(context_store)

        errors: list[Exception] = []

        def run_merge(ds_id: str, type_name: str) -> None:
            try:
                ctx = context_store.DatasetContext(
                    dataset_id=ds_id,
                    node_types=[
                        context_store.DatasetNodeType(name=type_name, maps_to=type_name)
                    ],
                )
                context_store.merge_into_shared(ctx)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=run_merge, args=(f"ds{i}", f"Type{i}"))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Merge raised: {errors}"
        shared = context_store.load_shared_context()
        present = {nt.name for nt in shared.node_types}
        for i in range(5):
            assert f"Type{i}" in present, f"Type{i} was lost in concurrent merge"


# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------

class TestSchemaVersioning:
    def _reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)
        return context_store

    def test_save_stamps_current_dataset_schema_version(self, tmp_path, monkeypatch):
        """save_dataset_context always writes schema_version == DATASET_CONTEXT_SCHEMA_VERSION."""
        cs = self._reload(tmp_path, monkeypatch)
        ctx = cs.DatasetContext(dataset_id="ds1", schema_version=0)
        cs.save_dataset_context(ctx)
        loaded = cs.load_dataset_context("ds1")
        assert loaded.schema_version == cs.DATASET_CONTEXT_SCHEMA_VERSION

    def test_old_dataset_file_without_schema_version_loads_ok(self, tmp_path, monkeypatch):
        """A YAML without schema_version defaults to 0 and loads without error."""
        cs = self._reload(tmp_path, monkeypatch)
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets" / "old_ds.yaml").write_text(
            "dataset_id: old_ds\nsource_file: ''\n"
        )
        ctx = cs.load_dataset_context("old_ds")
        assert ctx is not None
        assert ctx.schema_version == 0

    def test_future_dataset_schema_version_raises_value_error(self, tmp_path, monkeypatch):
        """A dataset context with schema_version > current raises ValueError."""
        cs = self._reload(tmp_path, monkeypatch)
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets" / "future_ds.yaml").write_text(
            "dataset_id: future_ds\nschema_version: 999\n"
        )
        with pytest.raises(ValueError, match="schema_version"):
            cs.load_dataset_context("future_ds")

    def test_future_dataset_schema_version_error_mentions_versions(self, tmp_path, monkeypatch):
        """The ValueError message includes both file version and current version."""
        cs = self._reload(tmp_path, monkeypatch)
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets" / "fv_ds.yaml").write_text(
            "dataset_id: fv_ds\nschema_version: 999\n"
        )
        with pytest.raises(ValueError) as exc_info:
            cs.load_dataset_context("fv_ds")
        msg = str(exc_info.value)
        assert "999" in msg
        assert str(cs.DATASET_CONTEXT_SCHEMA_VERSION) in msg

    def test_shared_context_merge_stamps_schema_version(self, tmp_path, monkeypatch):
        """After merge_into_shared, load_shared_context returns schema_version == current."""
        cs = self._reload(tmp_path, monkeypatch)
        ctx = cs.DatasetContext(
            dataset_id="ds1",
            node_types=[cs.DatasetNodeType(name="TC", maps_to="TC")],
        )
        cs.merge_into_shared(ctx)
        sc = cs.load_shared_context()
        assert sc.schema_version == cs.SHARED_CONTEXT_SCHEMA_VERSION

    def test_old_shared_file_without_schema_version_loads_ok(self, tmp_path, monkeypatch):
        """A shared_context.yaml without schema_version defaults to 0 and loads without error."""
        cs = self._reload(tmp_path, monkeypatch)
        (tmp_path / "shared_context.yaml").write_text(
            "version: 2\nupdated_at: '2026-01-01'\nnode_types: []\n"
            "relationship_types: []\nstructural_patterns: []\n"
        )
        sc = cs.load_shared_context()
        assert sc.schema_version == 0

    def test_future_shared_schema_version_raises_value_error(self, tmp_path, monkeypatch):
        """A shared context with schema_version > current raises ValueError."""
        cs = self._reload(tmp_path, monkeypatch)
        (tmp_path / "shared_context.yaml").write_text(
            "version: 1\nschema_version: 999\n"
            "node_types: []\nrelationship_types: []\nstructural_patterns: []\n"
        )
        with pytest.raises(ValueError, match="schema_version"):
            cs.load_shared_context()


# ---------------------------------------------------------------------------
# Record hash store
# ---------------------------------------------------------------------------

class TestRecordHashStore:
    def _reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPH_PIPELINE_CONTEXT_DIR", str(tmp_path))
        import importlib
        from graph_pipeline import context_store
        importlib.reload(context_store)
        return context_store

    def test_load_missing_returns_empty_dict(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        result = cs.load_record_hashes("nonexistent_ds")
        assert result == {}

    def test_save_then_load_round_trip(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        hashes = {"tc-001": "abc123def456abcd", "tc-002": "1234567890abcdef"}
        cs.save_record_hashes("ds1", hashes)
        loaded = cs.load_record_hashes("ds1")
        assert loaded == hashes

    def test_save_creates_parent_directory(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        cs.save_record_hashes("new_ds", {"tc-001": "abc123def456abcd"})
        assert (tmp_path / "datasets" / "new_ds_hashes.json").exists()

    def test_corrupted_file_returns_empty_dict(self, tmp_path, monkeypatch):
        cs = self._reload(tmp_path, monkeypatch)
        (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "datasets" / "bad_ds_hashes.json").write_text("not valid json {{{{")
        result = cs.load_record_hashes("bad_ds")
        assert result == {}
