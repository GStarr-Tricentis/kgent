# Run 11 — Issue #12: Incremental ingest / hash-based change detection

## Problem

Every ingest re-extracts and re-writes all records even when the dataset hasn't changed.
Neo4j MERGE statements are idempotent so there is no data corruption, but the work is
wasted: LLM extraction calls fire for unchanged records, and every MERGE round-trips
through the network.

The dataset-level fingerprint (issue #2) already skips schema discovery when the type
distribution is unchanged. This change adds record-level hashing: a per-record SHA-256
digest is stored after each ingest, and subsequent runs only extract and write records
whose digest has changed (or that are new). Records missing from the current file are
silently dropped from the hash store on the next successful write.

**Known limitation**: deletions are not propagated to Neo4j. If a record is removed from
the source file, its node persists in the graph. A future cleanup pass is needed for that.

**Does not depend on run 10 (schema versioning)**. Hashes live in a separate JSON file;
`DatasetContext` is not modified.

## Files changed

```
graph_pipeline/sampler.py         — add compute_record_hashes
graph_pipeline/context_store.py   — add import json, _hash_store_path,
                                    load_record_hashes, save_record_hashes
scripts/ingest.py                 — add --full-ingest flag; hash filtering block;
                                    use ingest_records in step 5; save hashes after step 7
tests/test_sampler.py             — add TestComputeRecordHashes (7 tests)
tests/test_context_store.py       — add TestRecordHashStore (4 tests)
```

---

## Change 1 — `graph_pipeline/sampler.py`: add `compute_record_hashes`

`hashlib` and `json` are already imported (used by `compute_fingerprint`). Append after
`compute_fingerprint` (currently the last function in the file).

**Add** (at end of file):
```python
def compute_record_hashes(records: list[dict], id_field: str) -> dict[str, str]:
    """Return {record_id: digest} for every record that has id_field.

    Digest is a 16-char hex SHA-256 of the record's JSON with sorted keys.
    Records without id_field are silently skipped — consistent with extractor behaviour.
    record_id is always coerced to str to match YAML round-trip semantics.
    """
    hashes: dict[str, str] = {}
    for record in records:
        record_id = record.get(id_field)
        if record_id is None:
            continue
        digest = hashlib.sha256(
            json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
        hashes[str(record_id)] = digest
    return hashes
```

---

## Change 2 — `graph_pipeline/context_store.py`: add `import json` and hash store helpers

### 2a — Add `import json`

The current import block has `import fcntl` then `import os`. Insert `import json`
between them (alphabetical order).

**Before**:
```python
import fcntl
import os
```

**After**:
```python
import fcntl
import json
import os
```

### 2b — Add `_hash_store_path` after `_dataset_path`

**Before** (lines immediately after `_dataset_path`):
```python
def _dataset_path(dataset_id: str) -> Path:
    return _context_dir() / "datasets" / f"{dataset_id}.yaml"


# ---------------------------------------------------------------------------
# YAML helpers
```

**After**:
```python
def _dataset_path(dataset_id: str) -> Path:
    return _context_dir() / "datasets" / f"{dataset_id}.yaml"


def _hash_store_path(dataset_id: str) -> Path:
    return _context_dir() / "datasets" / f"{dataset_id}_hashes.json"


# ---------------------------------------------------------------------------
# YAML helpers
```

### 2c — Add `load_record_hashes` and `save_record_hashes` in the Public API section

Insert after `save_dataset_context` (currently the last function before `merge_into_shared`).

**Before**:
```python
def save_dataset_context(ctx: DatasetContext) -> None:
    path = _dataset_path(ctx.dataset_id)
    _save_yaml(path, ctx.model_dump())


def merge_into_shared(dataset_ctx: DatasetContext) -> SharedContext:
```

**After**:
```python
def save_dataset_context(ctx: DatasetContext) -> None:
    path = _dataset_path(ctx.dataset_id)
    _save_yaml(path, ctx.model_dump())


def load_record_hashes(dataset_id: str) -> dict[str, str]:
    """Load the per-record hash store for a dataset. Returns {} if no store exists or it is corrupt."""
    path = _hash_store_path(dataset_id)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_record_hashes(dataset_id: str, hashes: dict[str, str]) -> None:
    """Persist per-record hashes to disk, overwriting the previous store."""
    path = _hash_store_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, sort_keys=True)


def merge_into_shared(dataset_ctx: DatasetContext) -> SharedContext:
```

(Note: if `save_dataset_context` currently uses `_save_yaml` rather than the above form,
check the existing implementation and preserve it — only insert the two new functions
between `save_dataset_context` and `merge_into_shared`.)

---

## Change 3 — `scripts/ingest.py`: add `--full-ingest` flag

**Before** (in the argparse block, after `--force-rediscover`):
```python
    parser.add_argument("--force-rediscover", action="store_true",
                        help="Re-run schema discovery even if the dataset fingerprint is unchanged")
    parser.add_argument("--config", default="kgent/config/config.yaml")
```

**After**:
```python
    parser.add_argument("--force-rediscover", action="store_true",
                        help="Re-run schema discovery even if the dataset fingerprint is unchanged")
    parser.add_argument("--full-ingest", action="store_true",
                        help="Process all records regardless of per-record hash cache")
    parser.add_argument("--config", default="kgent/config/config.yaml")
```

---

## Change 4 — `scripts/ingest.py`: hash filtering block (insert between step 4 and step 5)

Insert the new block between the end of step 4 and the step 5 header comment.

**Before** (lines 219–222):
```python
    # -------------------------------------------------------------------------
    # Step 5: Extract
    # -------------------------------------------------------------------------
    _step(5, TOTAL_STEPS, "Extracting nodes and relationships...")
```

**After**:
```python
    # -------------------------------------------------------------------------
    # Incremental: filter to changed / new records
    # -------------------------------------------------------------------------
    from graph_pipeline.sampler import compute_record_hashes
    from graph_pipeline.context_store import load_record_hashes

    current_hashes = compute_record_hashes(records, dataset_ctx.id_field)

    if not args.full_ingest and current_hashes:
        stored_hashes = load_record_hashes(dataset_id)
        changed_ids = {
            rid for rid, h in current_hashes.items()
            if stored_hashes.get(rid) != h
        }
        ingest_records = [
            r for r in records
            if str(r.get(dataset_ctx.id_field, "")) in changed_ids
        ]
        unchanged_count = len(records) - len(ingest_records)
        if unchanged_count:
            _indent(
                f"{unchanged_count} unchanged record(s) skipped; "
                f"{len(ingest_records)} to process."
            )
        if not ingest_records:
            _indent("All records unchanged — nothing to ingest.")
            print("\nDone.")
            return
    else:
        ingest_records = records

    # -------------------------------------------------------------------------
    # Step 5: Extract
    # -------------------------------------------------------------------------
    _step(5, TOTAL_STEPS, "Extracting nodes and relationships...")
```

The guard `and current_hashes` handles datasets where no record has the id_field: the
hash dict is empty, incremental filtering is skipped, and all records are processed.

---

## Change 5 — `scripts/ingest.py`: step 5 uses `ingest_records`

**Before** (step 5 body):
```python
    from graph_pipeline.extractor import extract_all
    nodes, rels = await extract_all(records, dataset_ctx, shared_ctx, backend=backend)
    _indent(f"{len(nodes)} nodes, {len(rels)} relationships")
```

**After**:
```python
    from graph_pipeline.extractor import extract_all
    nodes, rels = await extract_all(ingest_records, dataset_ctx, shared_ctx, backend=backend)
    _indent(f"{len(nodes)} nodes, {len(rels)} relationships")
```

Only `ingest_records` changes; everything else in the extraction call is identical.

---

## Change 6 — `scripts/ingest.py`: save hashes after successful write in step 7

**Before** (end of the `if not args.dry_run:` block in step 7):
```python
        if fatal_errors:
            print("\nERROR: write errors occurred.", file=sys.stderr)
            await driver.close()
            sys.exit(1)
        await driver.close()
    else:
        _indent("(dry run — no data written)")
```

**After**:
```python
        if fatal_errors:
            print("\nERROR: write errors occurred.", file=sys.stderr)
            await driver.close()
            sys.exit(1)
        await driver.close()
        from graph_pipeline.context_store import save_record_hashes
        save_record_hashes(dataset_id, current_hashes)
    else:
        _indent("(dry run — no data written)")
```

Hashes are saved only on the clean-success path (after `sys.exit(1)` on fatal errors,
so failed records will be retried on next run). Dry-run does not write hashes.

---

## Tests — `tests/test_sampler.py`: add `TestComputeRecordHashes`

Append after `TestComputeFingerprint`.

```python
# ---------------------------------------------------------------------------
# compute_record_hashes
# ---------------------------------------------------------------------------

class TestComputeRecordHashes:
    def _make_records(self):
        return [
            {"uniqueId": "tc-001", "typeName": "TestCase", "name": "Login"},
            {"uniqueId": "tc-002", "typeName": "TestCase", "name": "Logout"},
        ]

    def test_returns_dict(self):
        from graph_pipeline.sampler import compute_record_hashes
        result = compute_record_hashes(self._make_records(), "uniqueId")
        assert isinstance(result, dict)

    def test_keys_are_record_ids(self):
        from graph_pipeline.sampler import compute_record_hashes
        result = compute_record_hashes(self._make_records(), "uniqueId")
        assert set(result.keys()) == {"tc-001", "tc-002"}

    def test_values_are_16_char_hex(self):
        from graph_pipeline.sampler import compute_record_hashes
        result = compute_record_hashes(self._make_records(), "uniqueId")
        for v in result.values():
            assert isinstance(v, str)
            assert len(v) == 16
            assert all(c in "0123456789abcdef" for c in v)

    def test_deterministic(self):
        from graph_pipeline.sampler import compute_record_hashes
        records = self._make_records()
        assert compute_record_hashes(records, "uniqueId") == compute_record_hashes(records, "uniqueId")

    def test_changed_record_produces_different_hash(self):
        from graph_pipeline.sampler import compute_record_hashes
        original = [{"uniqueId": "tc-001", "name": "Login"}]
        modified = [{"uniqueId": "tc-001", "name": "Login CHANGED"}]
        h_orig = compute_record_hashes(original, "uniqueId")["tc-001"]
        h_mod = compute_record_hashes(modified, "uniqueId")["tc-001"]
        assert h_orig != h_mod

    def test_records_without_id_field_skipped(self):
        from graph_pipeline.sampler import compute_record_hashes
        records = [
            {"uniqueId": "tc-001", "name": "Login"},
            {"name": "No ID here"},
        ]
        result = compute_record_hashes(records, "uniqueId")
        assert "tc-001" in result
        assert len(result) == 1

    def test_hash_independent_of_dict_key_insertion_order(self):
        from graph_pipeline.sampler import compute_record_hashes
        r1 = {"uniqueId": "tc-001", "name": "Login", "status": "active"}
        r2 = {"status": "active", "uniqueId": "tc-001", "name": "Login"}
        h1 = compute_record_hashes([r1], "uniqueId")["tc-001"]
        h2 = compute_record_hashes([r2], "uniqueId")["tc-001"]
        assert h1 == h2
```

---

## Tests — `tests/test_context_store.py`: add `TestRecordHashStore`

Append after `TestSchemaVersioning` (or after `TestMergeSharedLocking` if run 10 hasn't
landed yet — place it at the end of the file either way).

```python
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
```

---

## Definition of done

- [ ] `compute_record_hashes(records, id_field)` present in `graph_pipeline/sampler.py`; uses `sort_keys=True` and `hexdigest()[:16]`; records missing id_field are skipped; record_id is `str`-coerced
- [ ] `import json` added to `graph_pipeline/context_store.py`
- [ ] `_hash_store_path`, `load_record_hashes`, `save_record_hashes` present in `context_store.py`; store path is `{context_dir}/datasets/{dataset_id}_hashes.json`; `load_record_hashes` returns `{}` on missing or corrupt file
- [ ] `--full-ingest` flag present in argparse
- [ ] Hash filtering block inserted between step 4 and step 5; uses `ingest_records` downstream; early-returns with "Done." when nothing changed; `--full-ingest` and empty `current_hashes` both bypass filtering
- [ ] Step 5 uses `ingest_records` not `records`
- [ ] `save_record_hashes(dataset_id, current_hashes)` called after `await driver.close()` on the clean-success path only (not on fatal errors, not in dry-run)
- [ ] `pytest tests/test_sampler.py` passes (7 new tests in `TestComputeRecordHashes`)
- [ ] `pytest tests/test_context_store.py` passes (4 new tests in `TestRecordHashStore`)
