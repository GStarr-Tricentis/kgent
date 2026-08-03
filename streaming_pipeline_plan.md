# Streaming Pipeline — Implementation Plan

## Problem

Every loader returns `list[dict]`. `extract_all` returns `(list[Node], list[Relationship])`.
`write_all` receives both lists before any write happens. At peak — between extraction and
the first Neo4j write — the process holds:

| Structure | 100k records × 5 KB | Notes |
|---|---|---|
| Raw records | ~500 MB | loaded by `load_file()` |
| Extracted nodes | ~200 MB+ | `list[Node]` from `extract_all` |
| Extracted rels | ~100 MB+ | `list[Relationship]` from `extract_all` |
| Cypher statement list | ~100 MB | pre-built in `write_nodes` / `write_relationships` (fixed by UNWIND PR) |

Target: constant write-buffer memory (~batch_size nodes + rels at any point), plus
compact index structures that are O(N) in element count but not in record size.

---

## Why streaming is non-trivial

Two extraction rules require global knowledge before they can process any record:

**Rule 3+4 — hierarchy edges** need `explicit_nodes_by_name`: a map from every
explicit node's `name` property to its `(id, label)`. This is built from Rule 1
output, which means all records must have been seen before any hierarchy edge can
be resolved.

**Rule 6b — path FK relationships** need `path_value_index`: a map from every
value of a path field (e.g. `nodePath`) to the `node_id` of the record that owns
it. Again requires a full pass before any lookup can succeed.

A naive single-pass streamer silently drops all hierarchy edges and path FK rels.
The design below solves this with deferred resolution.

---

## Architecture: three passes

```
File
 │
 ├─ Pass 1 (PRE-SCAN)  ── streams ALL records once
 │    Builds: reservoir sample, fingerprint, record hashes,
 │            ingest_set, deleted_ids
 │    Memory: O(sample_size) + O(N) hashes (compact strings, ~6 bytes each)
 │
 │  [Schema discovery + human review — unchanged]
 │
 ├─ Pass 2 (INDEX BUILD)  ── streams INGEST_SET records once
 │    Builds: name_index (for hierarchy), path_value_index (for path FKs)
 │    Memory: O(|ingest_set|) compact dicts
 │
 └─ Pass 3 (EXTRACT + WRITE)  ── streams INGEST_SET records once
      Rules 1, 2, 5, 6: extract inline → WriteBuffer → flush every batch_size rows
      Rule 3+4: collect path_tasks (deferred, tiny structs)
      Rule 6b: resolve using path_value_index from Pass 2
      Rule 7: collect llm_buffer (deferred)

     [Post-stream deferred tasks]
      Hierarchy: resolve path_tasks using name_index → write edges
      LLM: process llm_buffer with semaphore → write results
```

**First run** (no prior dataset_ctx): 3 file reads.
**Subsequent runs** (dataset_ctx cached): still 3 file reads. Pass 1 is very cheap
(no JSON parsing of nested fields, just hashing). Future optimization: merge Pass 1
and Pass 2 into a single read for subsequent runs, using the cached ctx. Out of
scope for this PR.

---

## Phase 1 — Streaming loaders

### `graph_pipeline/loaders/base.py`

Add `stream` as an abstract method alongside `load`. Both must be implemented.

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator


class DataLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[dict]:
        """Load all records into memory. Keep for small files and tests."""

    @abstractmethod
    def stream(self, path: str) -> Iterator[dict]:
        """Yield records one at a time without loading the full file."""

    @abstractmethod
    def can_handle(self, path: str) -> bool:
        """Return True if this loader handles the given file."""
```

### `graph_pipeline/loaders/jsonl_loader.py`

`stream` reads line-by-line, skipping the header record, without building a list.

```python
def stream(self, path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("kind") == "export-dump-header":
                continue
            yield record
```

### `graph_pipeline/loaders/csv_loader.py`

`csv.DictReader` already iterates row-by-row. `stream` wraps it without the
accumulator list.

```python
def stream(self, path: str) -> Iterator[dict]:
    delimiter = "\t" if path.endswith(".tsv") else ","
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            yield {k: _coerce(v) for k, v in row.items()}
```

### `graph_pipeline/loaders/sql_loader.py`

Use `fetchmany` instead of `fetchall`.

```python
_CHUNK_SIZE = 1000

def stream(self, path: str) -> Iterator[dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            cursor.execute(f"SELECT * FROM {table}")  # noqa: S608
            while True:
                rows = cursor.fetchmany(_CHUNK_SIZE)
                if not rows:
                    break
                for row in rows:
                    yield {"_table": table, **dict(row)}
    finally:
        conn.close()
```

### `graph_pipeline/loaders/json_loader.py`

JSON has no natural streaming format. `stream` falls back to `load` and yields
from the resulting list. This preserves the O(N) memory for JSON files, which are
typically far smaller than JSONL dumps (the 122k-line production file is JSONL).
Add a comment noting that `ijson` could be added later for large JSON arrays.

```python
def stream(self, path: str) -> Iterator[dict]:
    # JSON has no streaming format; load fully then yield.
    # For large JSON arrays, consider adding the ijson dependency.
    yield from self.load(path)
```

### `graph_pipeline/loaders/__init__.py`

Add a top-level `stream()` function alongside `load()`.

```python
def stream(path: str) -> Iterator[dict]:
    """Detect format and stream records one at a time."""
    for loader in _LOADERS:
        if loader.can_handle(path):
            yield from loader.stream(path)
            return
    raise ValueError(f"No loader found for: {path}")
```

---

## Phase 2 — Pre-scan (Pass 1)

### New dataclass and function in `graph_pipeline/sampler.py`

Add a `PrescanResult` dataclass and a `prescan()` function. The existing
`sample_records`, `compute_fingerprint`, and `compute_record_hashes` functions are
kept unchanged — they continue to work on lists for tests and callers that already
have records in memory.

```python
from __future__ import annotations
import hashlib, json, random
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class PrescanResult:
    sample: list[dict]                      # reservoir sample, truncated for LLM
    fingerprint: str                        # 16-hex-char type-distribution fingerprint
    current_hashes: dict[str, str]          # {record_id: sha256[:16]}
    ingest_ids: set[str]                    # record IDs that are new or changed
    deleted_ids: set[str]                   # record IDs present in stored hashes but not in file
    type_field: str | None                  # detected type discriminator field
    type_counts: Counter                    # {type_name: count} across all records
    total_records: int


def prescan(
    records_iter: Iterator[dict],
    id_field: str,
    stored_hashes: dict[str, str],
    sample_size: int = 50,
) -> PrescanResult:
    """Single-pass pre-scan: sample, fingerprint, hash, and diff — without loading.

    Uses reservoir sampling (Algorithm R) rather than stratified sampling.
    The sample is used for schema discovery where diversity matters more than
    exact proportionality.
    """
    reservoir: list[dict] = []
    current_hashes: dict[str, str] = {}
    type_counts: Counter = Counter()
    type_field_detected: str | None = None
    total = 0

    for record in records_iter:
        total += 1

        # Reservoir sampling (Algorithm R)
        if len(reservoir) < sample_size:
            reservoir.append(_truncate_nested_arrays(record))
        else:
            j = random.randint(0, total - 1)
            if j < sample_size:
                reservoir[j] = _truncate_nested_arrays(record)

        # Detect type field on first record that has one
        if type_field_detected is None:
            for candidate in _TYPE_FIELD_CANDIDATES:
                if candidate in record:
                    type_field_detected = candidate
                    break

        if type_field_detected and type_field_detected in record:
            type_counts[record[type_field_detected]] += 1

        # Record hash
        record_id = record.get(id_field)
        if record_id is not None:
            digest = hashlib.sha256(
                json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()[:16]
            current_hashes[str(record_id)] = digest

    # Fingerprint from type distribution + key set of sample
    all_keys_in_sample = sorted({k for r in reservoir for k in r.keys()})
    fp_payload = json.dumps(
        {"types": dict(type_counts), "keys": all_keys_in_sample}, sort_keys=True
    )
    fingerprint = hashlib.sha256(fp_payload.encode()).hexdigest()[:16]

    ingest_ids = {
        rid for rid, h in current_hashes.items()
        if stored_hashes.get(rid) != h
    }
    deleted_ids = set(stored_hashes.keys()) - set(current_hashes.keys())

    return PrescanResult(
        sample=reservoir,
        fingerprint=fingerprint,
        current_hashes=current_hashes,
        ingest_ids=ingest_ids,
        deleted_ids=deleted_ids,
        type_field=type_field_detected,
        type_counts=type_counts,
        total_records=total,
    )
```

---

## Phase 3 — Index build (Pass 2)

### New function in `graph_pipeline/extractor.py`

A lightweight streaming pass that builds the two indices needed for deferred
resolution. Runs only over the ingest_set, not the full file (see Limitations
below).

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator

from graph_pipeline.context_store import DatasetContext


@dataclass
class ExtractionIndices:
    # For Rule 3+4: {node_name: (namespaced_node_id, label)}
    name_to_node: dict[str, tuple[str, str]] = field(default_factory=dict)
    # For Rule 6b: {target_field: {field_value: namespaced_node_id}}
    path_value_index: dict[str, dict[str, str]] = field(default_factory=dict)


def build_extraction_indices(
    records_iter: Iterator[dict],
    dataset_ctx: DatasetContext,
) -> ExtractionIndices:
    """Pass 2: stream ingest records to build lookup indices for deferred rules.

    Runs in O(N) time and O(N) memory in index entries (not record size).
    Must complete before Pass 3 (extract_stream) begins.
    """
    indices = ExtractionIndices()
    dataset_id = dataset_ctx.dataset_id
    id_field = dataset_ctx.id_field
    type_field = dataset_ctx.type_field
    type_map = _node_type_map(dataset_ctx)

    # Initialise path_value_index keys from config
    for pfk in dataset_ctx.path_fk_relationships:
        indices.path_value_index.setdefault(pfk.target_field, {})

    for record in records_iter:
        uid = record.get(id_field)
        type_name = record.get(type_field)
        if not uid or not type_name:
            continue

        label = type_map.get(type_name, type_name)
        namespaced_id = f"{dataset_id}:{uid}"

        # name_to_node: used by hierarchy resolver
        name = record.get("name")
        if name:
            indices.name_to_node[str(name)] = (namespaced_id, label)

        # path_value_index: used by Rule 6b
        for target_field in indices.path_value_index:
            val = record.get(target_field)
            if val is not None:
                indices.path_value_index[target_field][str(val)] = namespaced_id

    return indices
```

**Limitation:** Both indices are built from the ingest_set only, not the full
dataset. This means a path FK from a *changed* record to an *unchanged* record
will fail to resolve in the index and be silently skipped (same as today when the
unchanged record is absent from the batch). The existing `check_referential_integrity`
Neo4j lookup catches these as post-write warnings. This is an acceptable trade-off
for the first streaming PR; building path indices from the full file (at the cost
of one extra pre-scan pass) can be added later.

---

## Phase 4 — Streaming extract + write (Pass 3)

### `WriteBuffer` helper in `graph_pipeline/neo4j_writer.py`

A small class that accumulates nodes and rels and flushes to Neo4j when the buffer
is full. Keeps Pass 3 from needing to manage batching manually.

```python
class WriteBuffer:
    """Accumulate extracted nodes and relationships; flush to Neo4j in batches.

    Must be used as an async context manager. A single Neo4j session is opened
    on entry and closed on exit, eliminating the per-flush session open/close
    overhead that would occur with 100k+ records at batch_size=500.

        async with WriteBuffer(driver, batch_size=500) as buf:
            await extract_and_write_stream(..., buf)
        # session closed; buf.result holds final counts
    """

    def __init__(self, driver, batch_size: int = 500):
        self._driver = driver
        self._batch_size = batch_size
        self._nodes: list[Node] = []
        self._rels: list[Relationship] = []
        self._session = None
        self.result = WriteResult()

    async def __aenter__(self) -> WriteBuffer:
        self._session = self._driver.session()
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if exc_type is None:
                await self.flush_all()
        finally:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)
            self._session = None

    async def add_node(self, node: Node) -> None:
        self._nodes.append(node)
        if len(self._nodes) >= self._batch_size:
            await self._flush_nodes()

    async def add_rel(self, rel: Relationship) -> None:
        self._rels.append(rel)
        if len(self._rels) >= self._batch_size:
            await self._flush_rels()

    async def flush_all(self) -> None:
        """Flush any remaining buffered nodes and relationships."""
        if self._nodes:
            await self._flush_nodes()
        if self._rels:
            await self._flush_rels()

    async def _flush_nodes(self) -> None:
        r = await _write_nodes_to_session(self._nodes, self._session, self._batch_size)
        self.result.merge(r)
        self._nodes = []

    async def _flush_rels(self) -> None:
        r = await _write_rels_to_session(self._rels, self._session, self._batch_size)
        self.result.merge(r)
        self._rels = []
```

`WriteBuffer` calls two private session-scoped helpers instead of delegating to
`write_nodes` / `write_relationships` (which each open their own session). Add
these alongside `WriteBuffer` in `neo4j_writer.py`:

```python
async def _write_nodes_to_session(
    nodes: list[Node],
    session,
    batch_size: int = 500,
) -> WriteResult:
    """Write nodes via an already-open session. Used by WriteBuffer."""
    result = WriteResult()
    if not nodes:
        return result
    by_label: dict[str, list[Node]] = {}
    for node in nodes:
        by_label.setdefault(node.label, []).append(node)
    batch_index = 0
    for label, label_nodes in by_label.items():
        cypher = generate_node_merge_batch(label)
        for i in range(0, len(label_nodes), batch_size):
            chunk = label_nodes[i : i + batch_size]
            params = {
                "rows": [
                    {
                        "id": n.id,
                        "props": n.properties,
                        "extraction_source": n.extraction_source.value,
                    }
                    for n in chunk
                ]
            }
            await _run_batch(
                session, [(cypher, params)], batch_index, result,
                count_key_created="nodes_created",
                count_key_matched="nodes_matched",
            )
            batch_index += 1
    return result


async def _write_rels_to_session(
    rels: list[Relationship],
    session,
    batch_size: int = 500,
) -> WriteResult:
    """Write relationships via an already-open session. Used by WriteBuffer."""
    result = WriteResult()
    if not rels:
        return result
    valid_rels = []
    for r in rels:
        if not r.from_label or not r.to_label:
            msg = f"Skipping relationship {r.type} ({r.from_id} -> {r.to_id}): missing label"
            logger.warning(msg)
            result.errors.append(msg)
        else:
            valid_rels.append(r)
    if not valid_rels:
        return result
    by_triple: dict[tuple[str, str, str], list[Relationship]] = {}
    for r in valid_rels:
        by_triple.setdefault((r.from_label, r.to_label, r.type), []).append(r)
    batch_index = 0
    for (from_label, to_label, rel_type), triple_rels in by_triple.items():
        cypher = generate_relationship_merge_batch(from_label, to_label, rel_type)
        for i in range(0, len(triple_rels), batch_size):
            chunk = triple_rels[i : i + batch_size]
            params = {
                "rows": [
                    {
                        "from_id": r.from_id,
                        "to_id": r.to_id,
                        "extraction_source": r.extraction_source.value,
                    }
                    for r in chunk
                ]
            }
            await _run_batch(
                session, [(cypher, params)], batch_index, result,
                count_key_created="relationships_created",
                count_key_matched="relationships_matched",
            )
            batch_index += 1
    return result
```

### New public API in `graph_pipeline/extractor.py`

`extract_and_write_stream` replaces `extract_all` in the streaming path. The
existing `extract_all` is kept unchanged for backward compatibility and tests.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator

from graph_pipeline.neo4j_writer import WriteBuffer, WriteResult


@dataclass
class StreamExtractResult:
    write_result: WriteResult
    # Deferred: (record_id, path_string, leaf_uid) — one per record with a path field
    path_tasks: list[tuple[str, str, str]] = field(default_factory=list)
    # Deferred: records containing ambiguous fields for LLM Rule 7
    llm_buffer: list[dict] = field(default_factory=list)


async def extract_and_write_stream(
    records_iter: Iterator[dict],
    dataset_ctx: DatasetContext,
    shared_ctx: SharedContext | None,
    indices: ExtractionIndices,
    buffer: WriteBuffer,
) -> StreamExtractResult:
    """Pass 3: stream ingest records, apply Rules 1/2/5/6/6b inline, defer 3+4 and 7.

    Rules 3+4 (hierarchy) and Rule 7 (LLM) cannot run inline:
    - Rule 3+4 needs the complete name_to_node index, which is now pre-built in
      ExtractionIndices from Pass 2, so hierarchy edges CAN be emitted inline —
      see implementation note below.
    - Rule 7 requires LLM calls; it is deferred to avoid blocking the write pipeline.

    Implementation note on hierarchy: since ExtractionIndices.name_to_node is
    already complete (built in Pass 2), hierarchy edges CAN be resolved inline
    during Pass 3. The path_tasks list in StreamExtractResult is kept for the
    case where a dataset does not run Pass 2 (e.g. dry-run without a driver).
    When indices.name_to_node is populated, hierarchy is resolved inline and
    path_tasks will be empty.
    """
    dataset_id = dataset_ctx.dataset_id
    id_field = dataset_ctx.id_field
    type_field = dataset_ctx.type_field
    type_map = _node_type_map(dataset_ctx)
    rel_map = _rel_type_map(dataset_ctx)
    rel_label_map = _rel_label_map(dataset_ctx)
    result = StreamExtractResult(write_result=buffer.result)
    phantom_nodes_seen: dict[str, bool] = {}

    for record in records_iter:
        uid = record.get(id_field)
        type_name = record.get(type_field)

        # Rule 1: primary node
        if uid and type_name:
            label = type_map.get(type_name, type_name)
            await buffer.add_node(Node(
                id=f"{dataset_id}:{uid}",
                label=label,
                properties={
                    **_resolve_property_paths(record, dataset_ctx.property_paths),
                    **_scalar_properties(record),
                },
                source_record_id=uid,
                extraction_source=ExtractionSource.RULE_BASED,
            ))

        # Rule 2: nested collections
        if uid:
            parent_label = type_map.get(type_name, type_name) if type_name else ""
            for nc in dataset_ctx.nested_collections:
                items = _get_nested(record, nc.field)
                if not isinstance(items, list):
                    continue
                for item in items:
                    child_uid = item.get(nc.id_field)
                    if not child_uid:
                        continue
                    await buffer.add_node(Node(
                        id=f"{dataset_id}:{child_uid}",
                        label=nc.child_label,
                        properties={k: v for k, v in item.items() if not isinstance(v, (dict, list))},
                        source_record_id=uid,
                        extraction_source=ExtractionSource.RULE_BASED,
                    ))
                    await buffer.add_rel(Relationship(
                        from_id=f"{dataset_id}:{uid}",
                        to_id=f"{dataset_id}:{child_uid}",
                        from_label=parent_label,
                        to_label=nc.child_label,
                        type=nc.edge_type,
                        properties={},
                        source_record_id=uid,
                        extraction_source=ExtractionSource.RULE_BASED,
                    ))

        # Rules 3+4: hierarchy — inline when index is available
        if uid and dataset_ctx.hierarchy_config is not None:
            cfg = dataset_ctx.hierarchy_config
            path = record.get(cfg.field, "")
            if path:
                segments = [s.strip() for s in path.split(cfg.separator) if s.strip()]
                if len(segments) >= 2:
                    if indices.name_to_node:
                        # Resolve inline using pre-built index
                        await _emit_hierarchy_inline(
                            segments, uid, dataset_id, cfg, indices.name_to_node, buffer,
                            phantom_nodes_seen,
                        )
                    else:
                        # Defer for post-stream resolution
                        result.path_tasks.append((uid, path, str(uid)))

        # Rule 5: associations
        ac = dataset_ctx.association_config
        if ac is not None and uid:
            this_label = type_map.get(type_name, type_name) if type_name else ""
            this_id = f"{dataset_id}:{uid}"
            for assoc in record.get(ac.array_field, []):
                edge_name = assoc.get(ac.edge_name_subfield, "")
                partner_id_raw = assoc.get(ac.partner_id_subfield)
                direction = assoc.get(ac.direction_subfield, ac.direction_default) if ac.direction_subfield else ac.direction_default
                if partner_id_raw is None or str(partner_id_raw).strip() == "":
                    continue
                canonical_type = rel_map.get(edge_name)
                if not canonical_type:
                    continue
                from_label, to_label = rel_label_map.get(edge_name, ("", ""))
                partner_id = f"{dataset_id}:{partner_id_raw}"
                from_id, to_id = (this_id, partner_id) if direction == "out" else (partner_id, this_id)
                await buffer.add_rel(Relationship(
                    from_id=from_id, to_id=to_id,
                    from_label=from_label, to_label=to_label,
                    type=canonical_type, properties={},
                    source_record_id=uid,
                    extraction_source=ExtractionSource.RULE_BASED,
                ))

        # Rule 6: implicit FKs
        for ir in dataset_ctx.implicit_relationships:
            fk_value = record.get(ir.edge_name)
            if not uid or fk_value is None:
                continue
            this_label = type_map.get(type_name, type_name) if type_name else ""
            target_ds = ir.target_dataset_id if ir.cross_dataset else dataset_id
            await buffer.add_rel(Relationship(
                from_id=f"{dataset_id}:{uid}",
                to_id=f"{target_ds}:{fk_value}",
                from_label=this_label or ir.from_type,
                to_label=ir.to_type,
                type=ir.maps_to, properties={},
                source_record_id=uid,
                extraction_source=ExtractionSource.RULE_BASED,
            ))

        # Rule 6b: path FKs
        for pfk in dataset_ctx.path_fk_relationships:
            index = indices.path_value_index.get(pfk.target_field, {})
            if not uid:
                continue
            this_label = type_map.get(type_name, type_name) if type_name else ""
            from_id = f"{dataset_id}:{uid}"
            if pfk.container_path is None:
                fk_value = record.get(pfk.fk_field)
                if fk_value:
                    to_id = index.get(str(fk_value))
                    if to_id:
                        await buffer.add_rel(Relationship(
                            from_id=from_id, to_id=to_id,
                            from_label=pfk.from_type or this_label, to_label=pfk.to_type,
                            type=pfk.maps_to, properties={},
                            source_record_id=uid,
                            extraction_source=ExtractionSource.RULE_BASED,
                        ))
            else:
                container = _get_nested(record, pfk.container_path)
                if isinstance(container, list):
                    for item in container:
                        if isinstance(item, dict):
                            fk_value = item.get(pfk.fk_field)
                            if fk_value:
                                to_id = index.get(str(fk_value))
                                if to_id:
                                    await buffer.add_rel(Relationship(
                                        from_id=from_id, to_id=to_id,
                                        from_label=pfk.from_type or this_label, to_label=pfk.to_type,
                                        type=pfk.maps_to, properties={},
                                        source_record_id=uid,
                                        extraction_source=ExtractionSource.RULE_BASED,
                                    ))

        # Rule 7: collect LLM-eligible records for deferred processing
        if dataset_ctx.ambiguous_fields and any(f in record for f in dataset_ctx.ambiguous_fields):
            result.llm_buffer.append(record)

    await buffer.flush_all()
    return result
```

### `_emit_hierarchy_inline` helper

Extracted from the existing `_build_hierarchy_structures` logic, adapted to use
the pre-built `name_to_node` index and emit to a `WriteBuffer` instead of
accumulating lists.

```python
async def _emit_hierarchy_inline(
    segments: list[str],
    leaf_uid: str,
    dataset_id: str,
    config: HierarchyConfig,
    name_to_node: dict[str, tuple[str, str]],
    buffer: WriteBuffer,
    phantom_nodes_seen: dict[str, bool],   # shared across calls, prevents duplicate phantom writes
) -> None:
    """Emit hierarchy nodes and edges inline during the streaming pass."""
    for i in range(len(segments) - 1):
        parent_seg = segments[i]
        child_seg = segments[i + 1]
        is_leaf = (i + 1 == len(segments) - 1)

        # Resolve parent
        if parent_seg in name_to_node:
            parent_id, parent_label = name_to_node[parent_seg]
        else:
            parent_id = f"{dataset_id}:path:{parent_seg}"
            parent_label = config.phantom_label
            if parent_id not in phantom_nodes_seen:
                phantom_nodes_seen[parent_id] = True
                await buffer.add_node(Node(
                    id=parent_id, label=config.phantom_label,
                    properties={"name": parent_seg}, source_record_id="",
                    extraction_source=ExtractionSource.PHANTOM,
                ))

        # Resolve child
        if is_leaf:
            child_id = f"{dataset_id}:{leaf_uid}"
            # Look up label from name_to_node by ID (index is keyed by name, not ID)
            # Fall back to phantom_label if not found — hierarchy extractor does the same
            child_label = next(
                (lbl for (nid, lbl) in name_to_node.values() if nid == child_id),
                config.phantom_label,
            )
        elif child_seg in name_to_node:
            child_id, child_label = name_to_node[child_seg]
        else:
            child_id = f"{dataset_id}:path:{child_seg}"
            child_label = config.phantom_label
            if child_id not in phantom_nodes_seen:
                phantom_nodes_seen[child_id] = True
                await buffer.add_node(Node(
                    id=child_id, label=config.phantom_label,
                    properties={"name": child_seg}, source_record_id="",
                    extraction_source=ExtractionSource.PHANTOM,
                ))

        await buffer.add_rel(Relationship(
            from_id=parent_id, to_id=child_id,
            from_label=parent_label, to_label=child_label,
            type=config.edge_type, properties={},
            source_record_id=leaf_uid,
            extraction_source=ExtractionSource.RULE_BASED,
        ))
```

Note: the leaf child-label lookup in `_emit_hierarchy_inline` is still a linear
scan over `name_to_node.values()`. This can be resolved by extending
`ExtractionIndices` with an `id_to_label: dict[str, str]` field in a follow-up,
mirroring Fix 3 from the quick-fixes PR.

### Deferred LLM extraction (post-Pass-3)

The existing `_llm_extract_ambiguous` function is reused. After Pass 3 completes,
call it with `result.llm_buffer`, then write the returned nodes and rels through
the same `WriteBuffer`.

```python
if dataset_ctx.ambiguous_fields and backend is not None and result.llm_buffer:
    llm_nodes, llm_rels = await _llm_extract_ambiguous(
        result.llm_buffer, dataset_ctx, type_map, backend
    )
    for node in llm_nodes:
        await buffer.add_node(node)
    for rel in llm_rels:
        await buffer.add_rel(rel)
    await buffer.flush_all()
```

---

## Phase 5 — `scripts/ingest.py` orchestration

Replace the current Steps 1 and 5 with the three-pass design. Steps 2–4 and 6–8
are unchanged.

**Step 1 (was: load + sample)** — Pre-scan:
```python
from graph_pipeline.loaders import stream as stream_file
from graph_pipeline.sampler import prescan
from graph_pipeline.context_store import load_record_hashes

stored_hashes = load_record_hashes(dataset_id)
scan = prescan(
    stream_file(file_path),
    id_field=dataset_ctx_hint or "uniqueId",   # see note below
    stored_hashes=stored_hashes,
    sample_size=sample_size,
)
sample = scan.sample
fingerprint = scan.fingerprint
```

**Note on `id_field` in Pass 1:** `prescan` needs `id_field` to compute per-record
hashes, but `id_field` comes from `dataset_ctx` which isn't known until after schema
discovery. For first runs, `id_field` defaults to `"uniqueId"` (correct for Tosca).
For subsequent runs, load the prior `dataset_ctx` before Pass 1 and use its
`id_field`. This mirrors the current `compute_record_hashes` call which already
uses `dataset_ctx.id_field` — so subsequent runs are already correct; only first
runs use the default.

**Steps 2–4** (load shared context, schema discovery, human review) — unchanged.

**Between Steps 4 and 5** — Index build (Pass 2):
```python
from graph_pipeline.extractor import build_extraction_indices

indices = build_extraction_indices(
    (r for r in stream_file(file_path) if str(r.get(dataset_ctx.id_field, "")) in scan.ingest_ids),
    dataset_ctx,
)
```

**Step 5 (was: extract)** — Streaming extract + write:
```python
from graph_pipeline.extractor import extract_and_write_stream
from graph_pipeline.neo4j_writer import WriteBuffer

async with WriteBuffer(driver, batch_size=batch_size) as buffer:
    await extract_and_write_stream(
        (r for r in stream_file(file_path) if str(r.get(dataset_ctx.id_field, "")) in scan.ingest_ids),
        dataset_ctx,
        shared_ctx,
        indices,
        buffer,
    )
write_result = buffer.result
```

**Steps 6–8** — with two changes in the streaming path:

- **`check_referential_integrity` is dropped from the streaming path.** The UNWIND
  relationship template uses `MATCH (a:...) MATCH (b:...)` — if either MATCH finds
  no node for a given row, that row produces no output and the iteration is silently
  skipped. No transaction error is raised; the relationship is simply not created.
  Dangling endpoints show up as a lower `relationships_created` count in `WriteResult`
  rather than as explicit errors. The pre-write strip-and-log behavior from the
  batch path is not replicated in this PR. Operators should watch the
  `relationships_created` vs. input-rel-count delta in the output.

- **`spot_check` is dropped from the streaming path.** Nodes and rels are not
  retained after streaming. Replace the `spot_check` call with a print of
  `WriteResult` statistics — total nodes and rels created/matched. This serves the
  same human sanity-check purpose in a more structured form:

  ```python
  _indent(f"{write_result.nodes_created} nodes created, {write_result.nodes_matched} matched")
  _indent(f"{write_result.relationships_created} rels created, {write_result.relationships_matched} matched")
  if write_result.errors:
      for err in write_result.errors:
          _indent(f"  ⚠ {err}")
  ```

- `check_label_coverage` is unchanged — it takes `labels` from `dataset_ctx`, not
  the in-memory nodes list.

---

## Interface contracts: what changes, what stays

| Symbol | Status | Notes |
|---|---|---|
| `DataLoader.load()` | Unchanged | Kept for tests and small files |
| `DataLoader.stream()` | **New abstract method** | All four loaders must implement |
| `loaders.load()` | Unchanged | Still works for callers with small files |
| `loaders.stream()` | **New** | Used by the streaming pipeline |
| `sampler.prescan()` | **New** | Does not replace existing functions |
| `sampler.sample_records()` | Unchanged | Still used in dry-run / tests |
| `sampler.compute_record_hashes()` | Unchanged | Still used directly in tests |
| `extractor.extract_all()` | Unchanged | Kept for unit tests and dry-run |
| `extractor.build_extraction_indices()` | **New** | Pass 2 |
| `extractor.extract_and_write_stream()` | **New** | Pass 3 |
| `neo4j_writer.WriteBuffer` | **New class** | Async context manager; holds a single session across all flushes |
| `neo4j_writer._write_nodes_to_session()` | **New private** | Session-scoped node write; called by WriteBuffer |
| `neo4j_writer._write_rels_to_session()` | **New private** | Session-scoped rel write; called by WriteBuffer |
| `neo4j_writer.write_nodes()` | Unchanged | Kept for tests and non-streaming path |
| `neo4j_writer.write_relationships()` | Unchanged | Kept for tests and non-streaming path |
| `neo4j_writer.write_all()` | Unchanged | Kept for dry-run/tests |
| `scripts/ingest.py` | **Modified** | Orchestration only |

---

## Test strategy

### New unit tests

**Loaders** (`tests/test_loaders.py`) — add `test_stream_*` variants alongside
existing `test_load_*` tests for each loader. Assert that:
- `stream()` yields the same records as `load()` (set equality by record content)
- `stream()` does not load the full file (mock `open` with a generator that tracks
  how much has been read; assert it stops at the first record when only one is consumed)
- JSONL `stream()` skips the header record
- SQL `stream()` respects `_CHUNK_SIZE` (mock `fetchmany` to return one chunk then empty)

**Sampler prescan** (`tests/test_sampler.py`) — add:
- `test_prescan_sample_size_respected` — `len(result.sample) <= sample_size` for any input
- `test_prescan_identifies_changed_records` — hash of modified record appears in `ingest_ids`
- `test_prescan_identifies_deleted_records` — record in `stored_hashes` but not in iter → `deleted_ids`
- `test_prescan_empty_iter` — returns sensible empty defaults
- `test_prescan_fingerprint_stable` — same records yield same fingerprint
- `test_prescan_fingerprint_changes_on_type_distribution_change`

**Extraction indices** (`tests/test_extractor.py`) — add to a new `TestBuildExtractionIndices` class:
- `test_name_index_populated` — records with `name` field appear in `name_to_node`
- `test_path_value_index_populated` — records with path FK target field are indexed
- `test_records_without_id_skipped`

**Streaming extraction** (`tests/test_extractor.py`) — add `TestExtractAndWriteStream`:
- Mirror key tests from `TestRule1NodeExtraction` through `TestRule7LlmExtraction`
  using `extract_and_write_stream` with a mock `WriteBuffer` that captures calls
  instead of writing to Neo4j
- `test_phantom_nodes_emitted_inline_when_index_available`
- `test_path_tasks_populated_when_index_empty` (deferred hierarchy fallback)
- `test_llm_buffer_populated_for_ambiguous_records`
- `test_write_buffer_flushed_on_batch_boundary` — verifies `add_node` triggers a
  flush after `batch_size` nodes, keeping memory bounded

**WriteBuffer** (`tests/test_neo4j_writer.py`) — add:
- `test_write_buffer_flushes_on_node_batch_size`
- `test_write_buffer_flushes_on_rel_batch_size`
- `test_write_buffer_flush_all_writes_remainder`
- `test_write_buffer_accumulates_result`
- `test_write_buffer_single_session_across_flushes` — mock `_write_nodes_to_session`
  to capture the `session` argument; assert all calls share the same object
- `test_write_buffer_context_manager_closes_session_on_exit` — assert the session's
  `__aexit__` is called even when `extract_and_write_stream` raises

### Existing tests to run (no changes expected)

```bash
.venv/bin/pytest tests/test_extractor.py tests/test_loaders.py \
                 tests/test_sampler.py tests/test_neo4j_writer.py \
                 tests/test_cypher_generator.py tests/test_validator.py -v
```

All 138 current tests must continue to pass — the streaming path is purely
additive; `extract_all` and `write_all` are not modified.

---

## Merge order

1. **Loaders** — `stream()` on all four loaders + `loaders.stream()` top-level function.
   Self-contained, no dependencies. Add tests. (~200 lines)

2. **`WriteBuffer`** — new class in `neo4j_writer.py`, plus `_write_nodes_to_session`
   and `_write_rels_to_session` private helpers. No changes to existing public
   functions. Add tests. (~120 lines)

3. **`prescan()`** — new function in `sampler.py`. No changes to existing functions.
   Add tests. (~80 lines)

4. **`build_extraction_indices()`** — new function in `extractor.py`. No changes to
   existing functions. Add tests. (~60 lines)

5. **`extract_and_write_stream()`** and `_emit_hierarchy_inline()` — new functions in
   `extractor.py`. Largest single change. Add tests. (~200 lines)

6. **`scripts/ingest.py`** — wire up the three-pass orchestration. No new functions,
   just restructured flow. Verify manually with `--dry-run` against real JSONL data
   before merging. (~80 lines changed)

---

## Known limitations and follow-ups

- **Reservoir sampling is not stratified.** The existing `sample_records` guarantees
  ≥3 records per type; the new `prescan` uses simple Algorithm R. For schema discovery
  this is acceptable (LLM still sees diverse records) but worth noting. A streaming
  stratified sampler can be added later if schema quality regresses.

- **`name_to_node` leaf label lookup is O(N) per leaf** (see `_emit_hierarchy_inline`
  above). Fix: add `id_to_label: dict[str, str]` to `ExtractionIndices`. Same pattern
  as Fix 3 in the quick-fixes PR.

- **Path FK indices built from ingest_set only.** Cross-record path FKs from a
  changed record to an unchanged record will be silently skipped (caught post-write
  by referential integrity check). Full-file path FK indexing requires an extra pass
  and can be added as a follow-up.

- **JSON files still load fully.** The `JsonLoader.stream()` fallback preserves the
  existing behaviour. True streaming for large JSON arrays requires `ijson`.

- **`prescan` uses default `id_field="uniqueId"` on first run.** For non-Tosca
  datasets on their first ingestion, hashes will be keyed incorrectly until schema
  discovery sets the real `id_field`. The effect is that all records appear as
  "new" on the second run (hash cache miss), which is harmless — they are
  re-extracted and re-merged.
