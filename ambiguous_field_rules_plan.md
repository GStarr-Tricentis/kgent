# Plan: Option A — Schema-time ambiguous field resolution

## What we're doing

Currently: `ambiguous_fields` is a list of field names saved to the YAML. At ingest time, every
eligible record is sent to the LLM in batches. O(N) LLM calls.

After: Schema discovery runs an additional LLM call (Call 4) once, producing an
`AmbiguousFieldRule` per field that specifies how to split and match it. At ingest time, Rule 7
becomes a deterministic string-split + UID lookup. Zero LLM calls at ingest.

---

## Phase 1 — New config type in `context_store.py`

Add a new Pydantic model:

```python
class AmbiguousFieldRule(BaseModel):
    field: str          # field name in the record
    delimiter: str      # character to split on
    rel_type: str       # canonical rel type (must match a relationship_types maps_to)
    from_type: str = "" # from node label; empty string = use the record's own label
    to_type: str        # to node label
    direction: str = "out"  # "out" = this record → matched node, "in" = matched node → this record
```

Add one field to `DatasetContext`:

```python
ambiguous_field_rules: list[AmbiguousFieldRule] = Field(default_factory=list)
```

Keep `ambiguous_fields: list[str]` — it is used as input to Call 4 and for backwards-compat
warnings.

---

## Phase 2 — Fix `_filter_ambiguous_by_uid_coverage` for delimited values

The existing filter checks whole field values against `uid_set`. A field containing
`"uid1,uid2,uid3"` scores 0% and is dropped even if every token is a valid UID. This means the
most common ambiguous field pattern — a delimited list of references — is currently always
filtered out before Call 4 ever runs.

Update `_filter_ambiguous_by_uid_coverage` to try splitting on a priority list of common
delimiters (`","`, `"|"`, `";"`, `" "`) before falling back to the whole-value check. A field
passes if any split attempt achieves ≥50% token match rate. Return the winning delimiter
alongside each kept field name so Call 4 receives it as a strong hint rather than
rediscovering it from scratch.

The updated return type becomes `list[tuple[str, str]]` — `(field_name, delimiter)` — with
`""` as the delimiter for whole-value matches (single-UID fields). Update the call site in
`propose_dataset_context` accordingly.

---

## Phase 3 — Schema discovery Call 4: `_resolve_ambiguous_field_rules`

### New file: `graph_pipeline/prompts/schema_proposal_ambiguous_rules.txt`

The prompt shows:
- The surviving ambiguous field names and their candidate delimiters (from the updated filter)
- Up to 25 sample values per field (reuse `_build_field_value_matrix` output, filtered to the
  flagged fields only)
- Known node types and their labels
- Known relationship types (name, maps_to, from, to)
- A representative slice of actual record UIDs so the LLM can see what the tokens match

The prompt asks for, per field:
- `delimiter` — what character splits the value into tokens (the filter hint is shown; LLM
  confirms or corrects)
- `rel_type` — which known `maps_to` relationship type to create
- `from_type` — from node label (`""` means "use the record's own label at runtime")
- `to_type` — to node label
- `direction` — `"out"` or `"in"`

Returns a JSON array. Add `graph_pipeline/schemas/ambiguous_field_rules.json` for structured
output (same pattern as the other schema files in that directory).

### New function in `schema_discovery.py`

```python
async def _resolve_ambiguous_field_rules(
    ambiguous_fields: list[tuple[str, str]],  # (field_name, candidate_delimiter)
    sample: list[dict],
    node_types: list[DatasetNodeType],
    relationship_types: list[DatasetRelationshipType],
    id_field: str,
    backend: ModelBackend,
    max_retries: int,
) -> list[AmbiguousFieldRule]:
```

**Failure mode**: follow Call 3's pattern exactly — catch all exceptions after retries are
exhausted, log a warning, and return `[]`. A Call 4 failure degrades gracefully to "no
ambiguous field rules this run"; it must not crash schema discovery for a dataset whose
node/rel schema is otherwise valid.

**Validation**: after parsing the LLM response, build
`allowed_rel_types = {rt.maps_to for rt in relationship_types}` and filter:

```python
valid_rules = [r for r in parsed_rules if r.rel_type in allowed_rel_types]
dropped = len(parsed_rules) - len(valid_rules)
if dropped:
    logger.warning("Dropped %d ambiguous_field_rules with unknown rel_type", dropped)
```

This mirrors the filtering already done in `_llm_extract_batch` and prevents hallucinated
rel types from reaching Neo4j on every subsequent ingest.

Wire into `propose_dataset_context` immediately after `_filter_ambiguous_by_uid_coverage`:

```python
ambiguous_field_rules: list[AmbiguousFieldRule] = []
if ambiguous_fields:
    ambiguous_field_rules = await _resolve_ambiguous_field_rules(
        ambiguous_fields, sample, node_types, rel_types,
        id_field=..., backend=backend, max_retries=max_retries,
    )
```

Pass `ambiguous_field_rules=ambiguous_field_rules` to the `DatasetContext(...)` constructor.

**Important**: `_resolve_ambiguous_field_rules` is a no-op (returns `[]`) when `ambiguous_fields`
is empty — which is the common case. No extra LLM call for normal datasets.

---

## Phase 4 — Add `uid_set` to `ExtractionIndices`

The deterministic Rule 7 needs to check whether a split token is an actual record UID.
The updated filter confirmed ≥50% of tokens are raw UIDs — we need the full set at ingest time.

In `extractor.py`, add to `ExtractionIndices`:

```python
uid_set: set[str] = field(default_factory=set)  # raw (non-namespaced) UIDs
```

In `build_extraction_indices`, add one line per record:

```python
if uid:
    indices.uid_set.add(str(uid))
```

This is O(N) in entries already being processed — no extra pass needed.

---

## Phase 5 — Deterministic Rule 7 in `extractor.py`

### New helper `_apply_ambiguous_field_rules` (sync, no LLM)

```python
def _apply_ambiguous_field_rules(
    record: dict,
    dataset_id: str,
    uid: str,
    this_label: str,
    rules: list[AmbiguousFieldRule],
    uid_set: set[str],
) -> list[Relationship]:
```

Logic per rule:
1. `field_value = record.get(rule.field)` — skip if None or not a string
2. `tokens = [t.strip() for t in field_value.split(rule.delimiter) if t.strip()]`
3. For each token: if `token in uid_set` → construct a `Relationship`
4. `from_label = rule.from_type or this_label`
5. Direction `"out"` → `(f"{dataset_id}:{uid}", f"{dataset_id}:{token}")`, `"in"` → reversed

Note: this helper never produces new nodes — only rels between existing records. Non-matching
tokens (values not in `uid_set`) are silently skipped, which is correct: not every token in a
delimited field is guaranteed to be a UID in the current dump.

### In `extract_and_write_stream`

Replace the current Rule 7 accumulation:

```python
# OLD
if dataset_ctx.ambiguous_fields and any(f in record for f in dataset_ctx.ambiguous_fields):
    result.llm_buffer.append(record)
```

with:

```python
# NEW
if dataset_ctx.ambiguous_field_rules and uid:
    this_label = type_map.get(type_name, type_name) if type_name else ""
    pending_rels.extend(_apply_ambiguous_field_rules(
        record, dataset_id, uid, this_label,
        dataset_ctx.ambiguous_field_rules, indices.uid_set,
    ))
```

Remove the post-loop LLM block entirely:

```python
# DELETE
if dataset_ctx.ambiguous_fields and backend is not None and result.llm_buffer:
    llm_nodes, llm_rels = await _llm_extract_ambiguous(...)
    for node in llm_nodes:
        await buffer.add_node(node)
    for rel in llm_rels:
        await buffer.add_rel(rel)
    await buffer.flush_all()
```

The new deterministic rels go into `pending_rels` inline and are covered by the existing
post-loop two-phase flush with no further changes.

---

## Phase 6 — Remove dead code

**In `extractor.py`:**
- Delete `_llm_extract_batch`
- Delete `_llm_extract_ambiguous`
- Delete `_ENTITY_EXTRACTION_BATCH_PROMPT` and the `Path(...)` reference at module top
- Remove `llm_buffer` from `StreamExtractResult`
- Remove `backend: ModelBackend | None = None` parameter from `extract_and_write_stream`

**`extract_all`**: used in ~50 places in `tests/test_extractor.py` and nowhere in production
code — `ingest.py` uses `extract_and_write_stream` exclusively. Migrating those tests to the
streaming path would require mocking Neo4j for every test. Keep `extract_all` as a test-only
convenience function instead:

- Update its Rule 7 to the same deterministic path: if `dataset_ctx.ambiguous_field_rules` is
  non-empty, call `_apply_ambiguous_field_rules`; otherwise skip
- Remove the `backend` parameter from `extract_all` — nothing uses it for Rule 7 anymore
- `_llm_extract_ambiguous` and `_llm_extract_batch` are still deleted; `extract_all` no longer
  needs them

**File deletion:**
- `graph_pipeline/prompts/entity_extraction_batch.txt`

---

## Phase 7 — Backwards compatibility

For datasets with `ambiguous_fields` set but `ambiguous_field_rules` empty (existing saved
YAMLs), add a warning in `ingest.py` Step 5 before `extract_and_write_stream`:

```python
if dataset_ctx.ambiguous_fields and not dataset_ctx.ambiguous_field_rules:
    _indent(
        "⚠ ambiguous_fields present but no ambiguous_field_rules found. "
        "Run with --force-rediscover to resolve extraction rules. "
        "Ambiguous field extraction will be skipped this run."
    )
```

No fallback to per-record LLM — that is the behaviour being eliminated. The graph will be
missing those edges until the user re-runs with `--force-rediscover`.

---

## Phase 8 — Schema preview and YAML

`DatasetContext` is Pydantic — `save_dataset_context` and `load_dataset_context` will
automatically include `ambiguous_field_rules` in the YAML with no additional changes.

Update `_schema_preview` in `ingest.py` to render `ambiguous_field_rules` when present so that
`--dry-run` shows the inferred rules alongside the rest of the schema.

---

## Phase 9 — Tests

### `tests/test_schema_discovery.py`

New test `test_resolve_ambiguous_field_rules`:
- Mock backend returns a valid JSON array with one rule
- Assert the returned list contains one `AmbiguousFieldRule` with correct field/delimiter/rel_type
- Test the no-op case: empty `ambiguous_fields` → no LLM call, returns `[]`

### `tests/test_extractor.py`

Three changes:

1. **`build_extraction_indices` uid_set** — new test verifying that after processing a set of
   records, `indices.uid_set` contains the raw (non-namespaced) UIDs of each record

2. **`_apply_ambiguous_field_rules` unit test** — a record with a comma-delimited field, a
   `uid_set` containing some of the tokens; assert correct rels emitted for matching tokens and
   nothing for non-matching tokens or tokens not in the set

3. **`TestExtractAndWriteStream` Rule 7 update** — replace the existing `ambiguous_fields + mock
   LLM` setup with `ambiguous_field_rules` on the DatasetContext; verify rels appear in
   `buf.rels` without any backend (pass `backend=None`); verify the old `llm_buffer` field is
   gone from the result

---

## Schema version

No bump to `DATASET_CONTEXT_SCHEMA_VERSION`. The new `ambiguous_field_rules` field defaults to
`[]`, so existing YAMLs load without error and behave identically to before (the
backwards-compat warning in Phase 7 covers the case where rules need to be regenerated). A
version bump is only warranted when old YAMLs would fail to load or produce wrong behaviour —
neither applies here.

---

## Files changed

| File | Change |
|---|---|
| `graph_pipeline/context_store.py` | Add `AmbiguousFieldRule`; add field to `DatasetContext` |
| `graph_pipeline/schema_discovery.py` | Fix `_filter_ambiguous_by_uid_coverage` for delimited values; add `_resolve_ambiguous_field_rules`; wire into `propose_dataset_context` |
| `graph_pipeline/prompts/schema_proposal_ambiguous_rules.txt` | New prompt |
| `graph_pipeline/schemas/ambiguous_field_rules.json` | New JSON schema for structured output |
| `graph_pipeline/extractor.py` | Add `uid_set` to indices; add `_apply_ambiguous_field_rules`; update `extract_and_write_stream` and `extract_all`; remove LLM dead code; remove `backend` param from both functions |
| `graph_pipeline/prompts/entity_extraction_batch.txt` | Delete |
| `scripts/ingest.py` | Backwards-compat warning in Step 5; update `_schema_preview` |
| `tests/test_schema_discovery.py` | New tests for updated filter and Call 4 |
| `tests/test_extractor.py` | New/updated tests for uid_set, rule application, and streaming Rule 7 |
