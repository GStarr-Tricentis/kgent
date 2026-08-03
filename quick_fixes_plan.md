# Scalability Quick Fixes — Implementation Plan

Four self-contained changes. Each can be reviewed and merged independently.
Estimated total effort: 1–2 days.

---

## Fix 1 — Semaphore on LLM gather in `_llm_extract_ambiguous`

### Problem

[`extractor.py:279`](graph_pipeline/extractor.py) fires all LLM batches concurrently with no
cap:

```python
results = await asyncio.gather(
    *[_llm_extract_batch(b, ...) for b in batches],
    return_exceptions=True,
)
```

At `batch_size=10` with 100k eligible records that is 10 000 simultaneous requests.
This exhausts file descriptors, overloads a local Ollama server, and triggers
rate-limiting on cloud APIs.

### Change

Add an `asyncio.Semaphore` around each coroutine so at most `max_concurrency`
batches run at once.

**`graph_pipeline/extractor.py`** — `_llm_extract_ambiguous` (line 265):

```python
async def _llm_extract_ambiguous(
    records: list[dict],
    dataset_ctx: DatasetContext,
    type_map: dict[str, str],
    backend: ModelBackend,
    batch_size: int = 10,
    max_concurrency: int = 20,
) -> tuple[list[Node], list[Relationship]]:
    ambiguous = dataset_ctx.ambiguous_fields
    eligible = [r for r in records if any(f in r for f in ambiguous)]
    if not eligible:
        return [], []

    batches = [eligible[i : i + batch_size] for i in range(0, len(eligible), batch_size)]
    sem = asyncio.Semaphore(max_concurrency)

    async def _guarded(batch):
        async with sem:
            return await _llm_extract_batch(batch, dataset_ctx, type_map, backend)

    results = await asyncio.gather(*[_guarded(b) for b in batches], return_exceptions=True)
    ...
```

The `max_concurrency` default of 20 is a conservative starting point, not a
benchmarked value. Local Ollama is typically single-threaded; cloud APIs have
per-minute rate limits where 20 concurrent is already aggressive. The right
number varies by backend and would need profiling against a real workload.

Do not wire `max_concurrency` through config in this PR. The extractor backend
shares the agent backend and has no dedicated config section; adding a half-wired
config key would be worse than a named constant. Wire it through when someone
actually needs to tune it.

### Tests

**Add to `tests/test_extractor.py` in `TestLlmExtractAmbiguous`:**

```python
async def test_max_concurrency_limits_simultaneous_calls(self):
    """Never more than max_concurrency batches in flight at once."""
    import asyncio, json as _json
    from graph_pipeline.extractor import _llm_extract_ambiguous

    in_flight = {"current": 0, "peak": 0}

    class TrackingBackend:
        async def complete(self, messages, tools, response_format=None):
            from kgent.agent.types import ModelResponse
            in_flight["current"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
            await asyncio.sleep(0)          # yield so other coroutines can start
            in_flight["current"] -= 1
            content = _json.dumps([])
            return ModelResponse(content=content, tool_calls=[], finish_reason="stop",
                                 assistant_message={"role": "assistant", "content": content},
                                 raw=None)

    records = [
        {"uniqueId": f"tc-{i:03d}", "typeName": "TestCase", "category": "x"}
        for i in range(50)
    ]
    await _llm_extract_ambiguous(
        records, self._ctx(), self._type_map(),
        TrackingBackend(), batch_size=1, max_concurrency=5,
    )
    assert in_flight["peak"] <= 5
```

**Existing tests that must still pass (no changes needed):**
- `test_batching_reduces_llm_calls`
- `test_records_without_ambiguous_field_excluded`
- `test_failed_batch_does_not_prevent_other_batches`

---

## Fix 2 — UNWIND-based batch writes

### Problem

**[`cypher_generator.py:27`](graph_pipeline/cypher_generator.py)** generates one
`MATCH/MATCH/MERGE` statement per relationship. `write_relationships` pre-builds
the full statement list before batching:

```python
statements = [generate_relationship_merge(r) for r in rels]   # O(N) tuples
batches = [statements[i : i + batch_size] for i in range(0, len(statements), batch_size)]
```

Each batch transaction runs one Cypher statement per row. At 100k relationships
that is 100k round-trips within a single transaction. The UNWIND pattern reduces
this to one statement per `(from_label, to_label, rel_type)` group per batch,
regardless of how many rows are in the group.

Nodes have the same issue: `generate_node_merge` produces one statement per node.

### Change

#### `graph_pipeline/cypher_generator.py`

Add two new batch generators. Keep the existing per-row functions — they are used
in tests and as documentation of the Cypher shape.

```python
def generate_node_merge_batch(label: str) -> str:
    """UNWIND template for a homogeneous batch of nodes with the same label."""
    return (
        f"UNWIND $rows AS row\n"
        f"MERGE (n:{label} {{id: row.id}})\n"
        f"SET n += row.props\n"
        f"SET n.ingested_at = datetime()\n"
        f"SET n.extraction_source = row.extraction_source"
    )


def generate_relationship_merge_batch(from_label: str, to_label: str, rel_type: str) -> str:
    """UNWIND template for a homogeneous batch of relationships with the same type triple."""
    return (
        f"UNWIND $rows AS row\n"
        f"MATCH (a:{from_label} {{id: row.from_id}})\n"
        f"MATCH (b:{to_label} {{id: row.to_id}})\n"
        f"MERGE (a)-[r:{rel_type}]->(b)\n"
        f"SET r.extraction_source = row.extraction_source"
    )
```

#### `graph_pipeline/neo4j_writer.py`

Replace the statement-list pattern in `write_nodes` and `write_relationships` with
label/type-grouped UNWIND batches.

**`write_nodes`** — group by label, then UNWIND in `batch_size`-row chunks:

```python
async def write_nodes(nodes: list[Node], driver, batch_size: int = 500) -> WriteResult:
    result = WriteResult()
    if not nodes:
        return result

    by_label: dict[str, list[Node]] = {}
    for node in nodes:
        by_label.setdefault(node.label, []).append(node)

    async with driver.session() as session:
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
```

`_run_batch` already iterates `statements: list[tuple[str, dict]]` so passing a
single-element list `[(cypher, params)]` requires no changes to `_run_batch`.

**`write_relationships`** — group by `(from_label, to_label, type)`, then UNWIND:

```python
async def write_relationships(rels: list[Relationship], driver, batch_size: int = 500) -> WriteResult:
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

    async with driver.session() as session:
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

**Counter fix:** `_run_batch` currently infers `total_matched = len(statements) -
total_created`. With UNWIND that formula breaks — `len(statements) == 1` but the
batch can contain hundreds of rows. Update `_run_batch` to use the row count from
`params["rows"]` instead:

```python
# Inside _run_batch, after the statement loop:
# Before: total_matched = len(statements) - total_created
# After:
row_count = sum(len(params["rows"]) for _, params in statements)
total_matched = row_count - total_created
```

Use `params["rows"]` directly — no `isinstance` fallback, no `.get` default.
After this refactor every caller passes `{"rows": [...]}` batches. A `KeyError`
means a caller is broken and should fail loudly, not silently miscounting.

### Tests

**`tests/test_cypher_generator.py`** — add tests for the two new generators:

```python
def test_generate_node_merge_batch_contains_unwind():
    from graph_pipeline.cypher_generator import generate_node_merge_batch
    cypher = generate_node_merge_batch("TestCase")
    assert "UNWIND $rows AS row" in cypher
    assert "MERGE (n:TestCase {id: row.id})" in cypher
    assert "SET n += row.props" in cypher
    assert "SET n.ingested_at = datetime()" in cypher


def test_generate_relationship_merge_batch_contains_unwind():
    from graph_pipeline.cypher_generator import generate_relationship_merge_batch
    cypher = generate_relationship_merge_batch("TestCase", "Requirement", "COVERS")
    assert "UNWIND $rows AS row" in cypher
    assert "MATCH (a:TestCase {id: row.from_id})" in cypher
    assert "MATCH (b:Requirement {id: row.to_id})" in cypher
    assert "MERGE (a)-[r:COVERS]->(b)" in cypher
```

**Existing tests that must still pass (no changes needed):**
- All of `test_neo4j_writer.py` — the public API of `write_nodes`,
  `write_relationships`, and `write_all` is unchanged; only the internal Cypher
  shape changes.
- `test_cypher_generator.py` — the per-row functions `generate_node_merge` and
  `generate_relationship_merge` are untouched.

---

## Fix 3 — O(N²) hierarchy leaf lookup

### Problem

[`extractor.py:133`](graph_pipeline/extractor.py) resolves the leaf node of each
path segment with a linear scan:

```python
leaf_explicit = next(
    (n for n in explicit_nodes_by_name.values() if n.id == child_id), None
)
```

`explicit_nodes_by_name` is keyed by `node.properties["name"]`, so ID lookup
falls back to a full iteration over all explicit nodes. For 100k records with
deep hierarchy paths this runs O(records × path_depth) iterations.

### Change

**`graph_pipeline/extractor.py`** — build a secondary index by `node.id` once
before the loop, inside `_build_hierarchy_structures`:

```python
def _build_hierarchy_structures(
    records: list[dict],
    config: HierarchyConfig,
    id_field: str,
    dataset_id: str,
    explicit_nodes_by_name: dict[str, Node],
) -> tuple[dict[str, Node], list[Relationship]]:
    phantom_nodes: dict[str, Node] = {}
    hierarchy_rels: list[Relationship] = []

    # Build a secondary index for O(1) leaf resolution by node ID.
    explicit_nodes_by_id: dict[str, Node] = {n.id: n for n in explicit_nodes_by_name.values()}

    ...

    for record in records:
        ...
        for i in range(len(segments) - 1):
            ...
            if is_leaf:
                child_id = f"{dataset_id}:{record.get(id_field, '')}"
                leaf_explicit = explicit_nodes_by_id.get(child_id)   # O(1)
                child_label = leaf_explicit.label if leaf_explicit else config.phantom_label
            ...
```

This is a two-line net change (one `dict` comprehension added, one `next(...)`
replaced with `.get(...)`). The `explicit_nodes_by_id` dict is local to each
`_build_hierarchy_structures` call and is garbage-collected when the function
returns.

### Tests

No new tests required. The existing `TestRule3And4NodePath` suite covers all the
affected paths; particularly `test_explicit_record_used_as_intermediate_not_phantom`
and `test_contains_edge_connects_adjacent_segments` which exercise the leaf
resolution logic directly.

Run the full extractor suite after the change to confirm no regressions:

```bash
pytest tests/test_extractor.py -v
```

---

## Fix 4 — Label-scoped MATCH in `soft_delete_nodes` and `check_referential_integrity`

### Problem

Both functions issue a labelless `MATCH (n {id: id})` against Neo4j. Because `id`
is a property (not the internal Neo4j id), this forces a property index scan across
all labels rather than using the uniqueness constraint index that already exists per
label. Neither is batched.

**`neo4j_writer.py:184`:**
```python
"UNWIND $ids AS id MATCH (n {id: id}) SET n.deleted_at = datetime() RETURN count(n) AS cnt"
```

**`validator.py:89`:**
```python
"UNWIND $ids AS id MATCH (n {id: id}) RETURN n.id AS id"
```

### Change

#### `graph_pipeline/neo4j_writer.py` — `soft_delete_nodes`

Remove the unused `dataset_id` parameter (it never appears in the function body
and the name misleadingly implies dataset-scoped deletion). Add an optional
`labels` parameter. When labels are provided, run one indexed UNWIND per label
instead of one labelless scan over all IDs.

Update the call site in `scripts/ingest.py` accordingly (drop `dataset_id`,
pass `labels`).

```python
async def soft_delete_nodes(
    node_ids: list[str],
    driver,
    labels: list[str] | None = None,
) -> int:
    """Set deleted_at = datetime() on nodes whose id is in node_ids.

    Pass labels (the node labels in use for this dataset) to enable indexed
    lookups. Without labels, falls back to a labelless scan — correct but slower.
    """
    if not node_ids:
        return 0
    total = 0
    async with driver.session() as session:
        if labels:
            for label in labels:
                result = await session.run(
                    f"UNWIND $ids AS id MATCH (n:{label} {{id: id}}) "
                    f"SET n.deleted_at = datetime() RETURN count(n) AS cnt",
                    ids=node_ids,
                )
                record = await result.single()
                total += record["cnt"] if record else 0
        else:
            result = await session.run(
                "UNWIND $ids AS id MATCH (n {id: id}) SET n.deleted_at = datetime() RETURN count(n) AS cnt",
                ids=node_ids,
            )
            record = await result.single()
            total = record["cnt"] if record else 0
    return total
```

**Call site in `scripts/ingest.py`** — pass labels from the dataset context:

```python
# Existing call (approximate location, Step 7):
node_labels = [nt.maps_to for nt in dataset_ctx.node_types]
await soft_delete_nodes(pruned_ids, driver, labels=node_labels)
```

#### `graph_pipeline/validator.py` — `check_referential_integrity`

Group missing IDs by the label of the endpoint that references them, then query
per label using the uniqueness constraint index. Fall back to the labelless
query for any IDs whose label is unknown.

```python
async def check_referential_integrity(
    nodes: list[Node],
    relationships: list[Relationship],
    driver=None,
) -> list[ValidationError]:
    if not relationships:
        return []

    batch_ids: set[str] = {n.id for n in nodes}
    errors: list[ValidationError] = []

    # Collect missing IDs, preserving which label to use for indexed lookup.
    # from_id → from_label, to_id → to_label (first reference wins per id).
    missing_ids: dict[str, str] = {}          # id → source_record_id
    missing_id_labels: dict[str, str] = {}    # id → node label for indexed lookup

    for rel in relationships:
        for endpoint_id, label in ((rel.from_id, rel.from_label), (rel.to_id, rel.to_label)):
            if endpoint_id not in batch_ids and endpoint_id not in missing_ids:
                missing_ids[endpoint_id] = rel.source_record_id
                if label:
                    missing_id_labels[endpoint_id] = label

    if not missing_ids:
        return []

    if driver is None:
        for missing_id, record_id in missing_ids.items():
            errors.append(ValidationError(
                severity="warning",
                message=f"Endpoint '{missing_id}' not found in current batch (dry-run)",
                record_id=record_id,
                entity_id=missing_id,
            ))
        return errors

    # Group by label for indexed queries; collect any unknowns for a fallback scan.
    by_label: dict[str, list[str]] = {}
    unlabelled: list[str] = []
    for mid in missing_ids:
        label = missing_id_labels.get(mid)
        if label:
            by_label.setdefault(label, []).append(mid)
        else:
            unlabelled.append(mid)

    found_in_neo4j: set[str] = set()
    try:
        async with driver.session() as session:
            for label, ids in by_label.items():
                result = await session.run(
                    f"UNWIND $ids AS id MATCH (n:{label} {{id: id}}) RETURN n.id AS id",
                    ids=ids,
                )
                for record in await result.data():
                    found_in_neo4j.add(record["id"])

            if unlabelled:
                result = await session.run(
                    "UNWIND $ids AS id MATCH (n {id: id}) RETURN n.id AS id",
                    ids=unlabelled,
                )
                for record in await result.data():
                    found_in_neo4j.add(record["id"])
    except Exception as exc:
        logger.error("Neo4j lookup failed during referential integrity check: %s", exc)

    for missing_id, record_id in missing_ids.items():
        if missing_id not in found_in_neo4j:
            errors.append(ValidationError(
                severity="error",
                message=f"Endpoint '{missing_id}' not found in batch or in Neo4j",
                record_id=record_id,
                entity_id=missing_id,
            ))
    return errors
```

### Tests

**`soft_delete_nodes`** — add to `tests/integration/test_neo4j_writer.py` (or a
new unit test with a mock session):

```python
async def test_soft_delete_with_labels_uses_label_scoped_query(mocker):
    """When labels are provided, the MATCH clause includes the label."""
    session = AsyncMock()
    result_mock = AsyncMock()
    result_mock.single = AsyncMock(return_value={"cnt": 1})
    session.run = AsyncMock(return_value=result_mock)

    driver_mock = MagicMock()
    driver_mock.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver_mock.session.return_value.__aexit__ = AsyncMock(return_value=False)

    from graph_pipeline.neo4j_writer import soft_delete_nodes
    count = await soft_delete_nodes(["ds1:tc-001"], driver_mock, labels=["TestCase"])

    assert count == 1
    call_args = session.run.call_args[0][0]
    assert "TestCase" in call_args            # label present in MATCH
    assert "{id: id}" in call_args            # still using id property
```

**`check_referential_integrity`** — the rewrite has three new branches that need
explicit coverage. The existing dry-run and live-mode "all found / some missing"
tests remain valid regression coverage, but they do not exercise the new control
flow. Add all five tests below:

| Test | Branch covered |
|---|---|
| `test_all_labeled_uses_only_label_scoped_queries` | All endpoints have known labels → only label-scoped `MATCH (n:Label {id: id})` queries are fired; unlabelled fallback query is never called |
| `test_mixed_labeled_and_unlabeled_uses_both_paths` | Some endpoints have no label → unlabelled fallback is used for those IDs; labeled queries are still used for the others |
| `test_neo4j_exception_caught_returns_errors_for_all_unresolved` | Neo4j raises during the query loop → exception is caught and logged; function returns `ValidationError` entries for all unresolved IDs rather than re-raising |
| `test_all_endpoints_found_returns_no_errors` | (regression) All missing-from-batch IDs are found in Neo4j → empty error list |
| `test_missing_endpoint_returns_error` | (regression) An ID not in batch and not in Neo4j → `ValidationError` with severity "error" |

---

## Recommended merge order

1. **Fix 3** — 2-line change, zero API impact, zero test changes. Merge first.
2. **Fix 1** — Additive parameter `max_concurrency`, one new test. Low risk.
3. **Fix 4** — Additive parameter on `soft_delete_nodes`, internal refactor of
   `check_referential_integrity`. Straightforward but touches two files and the
   call site in `ingest.py`.
4. **Fix 2** — Most lines changed, most test coverage needed. Merge last so the
   others are already in when this one is reviewed.
