# Plan: Smarter schema discovery — nested collections + nested FKs + partner type validation

## What the current pipeline misses and why

**Nested collections (moduleAttributes)**: `summarize_structure` already identifies
nested array-of-object fields and injects them into the Call 1 prompt as "Nested
array-of-object fields". The nodes prompt already asks for `nested_collections` with a
correct example. The LLM has all the information — it just fails to reliably act on it
when the child objects don't have their own `typeName` (they're not typed records, just
sub-objects). Deterministic pre-scanning and explicitly handing candidates to the LLM
to confirm rather than discover fixes this.

**Nested uniqueId FKs (details.steps[*].moduleUniqueId)**: The relationships prompt has
a hard rule: "Implicit FK edge_name values must be top-level record fields." This
explicitly tells the LLM to ignore nested array FKs. The `path_fk_relationships` schema
already supports `target_field: uniqueId` (the default is nodePath, but it's just a
string) — the prompt just never teaches the LLM this case. Removing the top-level
restriction and adding a uniqueId-based path FK example fixes this.

**Association partner type heterogeneity (USES_MODULE → XModule + ApiModule)**: No
check exists. The LLM picks the most common partner type and sets `to_type` to it.
A deterministic post-Call-2 scan of the sample can detect mixed partner types and
clear `to_type` before the schema is saved.

---

## Phase 1 — Deterministic nested_collections pre-scan

### New helper: `_scan_nested_collection_candidates`

Add to `graph_pipeline/schema_discovery.py`:

```python
def _scan_nested_collection_candidates(
    sample: list[dict],
    id_field: str,
) -> list[dict]:
    """Scan sample records for array-of-objects fields whose items contain id_field.

    Returns a list of candidate dicts, each with:
      - field: dot-path to the array (e.g. "moduleAttributes" or "details.steps")
      - example_keys: sorted list of keys from the first matching item
      - records_with_field: count of sample records that have this field non-empty

    Scans top-level fields and one level deep into dict-valued fields (e.g. details.*).
    """
    candidates: dict[str, dict] = {}

    def _check_array(field_path: str, value) -> None:
        if not isinstance(value, list):
            return
        items_with_id = [
            item for item in value
            if isinstance(item, dict) and item.get(id_field) is not None
        ]
        if not items_with_id:
            return
        if field_path not in candidates:
            candidates[field_path] = {
                "field": field_path,
                "example_keys": sorted(items_with_id[0].keys()),
                "records_with_field": 0,
            }
        candidates[field_path]["records_with_field"] += 1

    for record in sample:
        for key, value in record.items():
            _check_array(key, value)
            # One level deep into dict-valued fields (e.g. details.moduleAttributes)
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    _check_array(f"{key}.{subkey}", subvalue)

    return list(candidates.values())
```

### Pass candidates into Call 1

In `propose_dataset_context`, call the scan before `_propose_node_types`:

```python
nested_candidates = _scan_nested_collection_candidates(
    sample, id_field=structural_config.get("id_field") or "uniqueId"
)
```

Wait — `structural_config` comes from `_propose_node_types`. Do the scan with a
pre-detected id_field first. `_detect_type_field` already runs before Call 1; add a
`_detect_id_field` helper (or default to `"uniqueId"`) for the pre-scan:

```python
# Pre-scan before Call 1 — uses heuristic id_field, confirmed by LLM output
_prescan_id_field = next(
    (
        k for k in {"uniqueId", "id", "uuid"}
        if any(k in r for r in sample)
    ),
    "uniqueId",
)
nested_candidates = _scan_nested_collection_candidates(sample, _prescan_id_field)
```

Pass `nested_candidates` as a new parameter to `_propose_node_types`. Inside that
function, serialize it and inject into the prompt as `{nested_collection_candidates}`.

### Update the nodes prompt

Add a new section between DATASET STRUCTURE SUMMARY and SAMPLE RECORDS:

```
PRE-IDENTIFIED NESTED COLLECTION CANDIDATES:
{nested_collection_candidates}

These fields contain arrays of objects where items have an ID field. Review each
candidate and include it in nested_collections if the items represent distinct entities
that deserve their own nodes (e.g. UI attributes, test steps with their own identity).
Exclude candidates where the array contains config blobs, primitive-only objects, or
items that are already modeled as top-level records.
```

Where `{nested_collection_candidates}` is `json.dumps(nested_candidates, indent=2)`,
or `"(none detected)"` when the list is empty.

**Why this works**: the LLM no longer has to discover candidates by reading raw JSON —
it gets a pre-filtered list of fields that have the right structure. Its job is
classification (entity array vs. config blob), which it does well.

---

## Phase 2 — Relationships prompt: nested uniqueId FKs

Two targeted changes to `graph_pipeline/prompts/schema_proposal_relationships.txt`.

### Change 1: rewrite the implicit_relationships rule

Current rule (line 69):
```
- Implicit FK edge_name values must be top-level record fields whose raw values
  directly match the id_field (typically uniqueId) of another record — fields ending
  in "UniqueId" or "Id" are the primary signal
```

Replace with:
```
- implicit_relationships are for top-level record fields whose raw values directly
  match the id_field of another record. Fields ending in "UniqueId" or "Id" are the
  primary signal. For the same type of reference found *inside* a nested array, use
  path_fk_relationships instead (see below).
```

### Change 2: expand path_fk_relationships to cover uniqueId-based nested FKs

Current description covers only nodePath-based FKs. Replace the entire
path_fk_relationships item in the TASK section (item 3) with:

```
3. "path_fk_relationships": fields inside nested arrays that reference other records,
   either by path string or by direct ID.

   Two sub-cases:

   a) PATH-BASED: the fk_field contains "/" -separated path strings matching another
      record's nodePath. Set target_field to "nodePath".
      Example: entries inside an ExecutionList have "testCaseNodePath" whose values
      match TestCase nodePath values.
      {"container_path": "details.entries", "fk_field": "testCaseNodePath",
       "target_field": "nodePath", "maps_to": "REFERENCES_TEST_CASE",
       "from_type": "ExecutionList", "to_type": "TestCase"}

   b) ID-BASED: the fk_field contains raw uniqueId values referencing another node.
      Fields ending in "UniqueId" or "Id" inside nested arrays are the signal.
      Set target_field to "uniqueId". Set to_type to "" if the partner could be
      multiple concrete types (e.g. both XModule and ApiModule).
      Example: each step in details.steps has "moduleUniqueId" pointing to a module.
      {"container_path": "details.steps", "fk_field": "moduleUniqueId",
       "target_field": "uniqueId", "maps_to": "STEP_USES_MODULE",
       "from_type": "", "to_type": ""}

   For each field listed under "Nested array-of-object fields" in the structure
   summary, inspect the sample to see if items have ID-looking fields. If yes,
   propose a path_fk_relationship.
```

Also update the path_fk_relationships schema example at the bottom of the prompt to
include the new fields:

```
"path_fk_relationships": [
  {"container_path": "<dot-path to array, or null>",
   "fk_field": "<field holding the path or ID>",
   "target_field": "<'nodePath' for path match, 'uniqueId' for direct ID match>",
   "maps_to": "<SCREAMING_SNAKE_CASE>",
   "from_type": "<canonical label of source, or '' to use record's own type>",
   "to_type": "<canonical label of target, or '' if multiple concrete types possible>"}
]
```

And update the final rules section to add:
```
- For each nested array field in the structure summary, check if items have fields
  ending in UniqueId or Id — if yes, propose a path_fk_relationship with
  target_field: "uniqueId"
- Set to_type to "" when a nested FK partner could be more than one concrete type
```

---

## Phase 3 — Deterministic association partner type validation

### New helper: `_validate_association_partner_types`

Add to `graph_pipeline/schema_discovery.py`:

```python
def _validate_association_partner_types(
    rel_types: list[DatasetRelationshipType],
    assoc_config: dict,
    sample: list[dict],
    id_field: str,
    type_field: str,
) -> list[DatasetRelationshipType]:
    """Clear to_type on association rules where sample partners have multiple concrete types.

    Builds a uid→type map from the sample, then for each relationship_type checks
    whether the actual partner types in the sample associations are heterogeneous.
    If a rule's to_type doesn't match the observed partners, clears it to "".
    """
    if not assoc_config or not rel_types:
        return rel_types

    array_field = assoc_config.get("array_field", "associations")
    edge_name_sub = assoc_config.get("edge_name_subfield", "edgeName")
    partner_id_sub = assoc_config.get("partner_id_subfield", "partnerUniqueId")

    # uid → typeName for all sample records
    uid_type: dict[str, str] = {}
    for record in sample:
        uid = record.get(id_field)
        tname = record.get(type_field)
        if uid is not None and tname is not None:
            uid_type[str(uid)] = str(tname)

    # edgeName → set of observed partner types
    edge_partner_types: dict[str, set[str]] = {}
    for record in sample:
        for assoc in record.get(array_field, []):
            if not isinstance(assoc, dict):
                continue
            edge_name = assoc.get(edge_name_sub)
            partner_id = assoc.get(partner_id_sub)
            if not edge_name or partner_id is None:
                continue
            ptype = uid_type.get(str(partner_id))
            if ptype:
                edge_partner_types.setdefault(edge_name, set()).add(ptype)

    updated: list[DatasetRelationshipType] = []
    for rt in rel_types:
        partner_types = edge_partner_types.get(rt.name, set())
        if len(partner_types) > 1:
            logger.warning(
                "association '%s' has heterogeneous partner types %s — "
                "clearing to_type (was '%s') to allow label-free MATCH",
                rt.name, sorted(partner_types), rt.to_type,
            )
            updated.append(rt.model_copy(update={"to_type": ""}))
        elif partner_types and rt.to_type:
            # Map typeName back to canonical label for comparison
            observed = next(iter(partner_types))
            # Compare loosely — typeName may equal maps_to directly
            if observed != rt.to_type:
                logger.warning(
                    "association '%s' to_type '%s' doesn't match observed partner "
                    "type '%s' in sample — clearing to_type",
                    rt.name, rt.to_type, observed,
                )
                updated.append(rt.model_copy(update={"to_type": ""}))
            else:
                updated.append(rt)
        else:
            updated.append(rt)

    return updated
```

### Wire into `propose_dataset_context`

After the `_propose_relationship_types` call returns, add:

```python
rel_types = _validate_association_partner_types(
    rel_types,
    assoc_config_dict,
    sample,
    id_field=structural_config.get("id_field") or "uniqueId",
    type_field=structural_config.get("type_field") or "typeName",
)
```

**Limitation**: the sample uid_type map only covers ~50 records. A minority partner
type (e.g. 8% ApiModule) has ~4 occurrences in a 50-record sample — likely enough to
catch, but not guaranteed. This is a best-effort check, not a guarantee.

---

## What each fix catches

| Gap | Fixed by |
|---|---|
| moduleAttributes (12k nodes) | Phase 1 — deterministic candidate → LLM confirm |
| details.steps[*].moduleUniqueId | Phase 2 — prompt teaches uniqueId-based path FK |
| USES_MODULE → ApiModule (80 rels) | Phase 3 — deterministic partner type check |
| Future heterogeneous associations | Phase 3 — generalises to any edgeName |
| Future nested ID FKs | Phase 2 — LLM now knows to look inside nested arrays |

---

## Runtime and memory impact

All three phases run during schema discovery (Step 3), which processes only the sample
(typically 50–500 records) and runs once on first ingest.

- Phase 1: O(sample_size × fields_per_record) dict inspection. Negligible.
- Phase 2: prompt text change only. Zero runtime cost.
- Phase 3: O(sample_size × associations_per_record) dict lookups. Negligible.

No impact on Pass 2, Pass 3a, or Pass 3b.

---

## Files changed

| File | Change |
|---|---|
| `graph_pipeline/schema_discovery.py` | Add `_scan_nested_collection_candidates`, `_validate_association_partner_types`; wire both into `propose_dataset_context`; pass candidates to `_propose_node_types` |
| `graph_pipeline/prompts/schema_proposal_nodes.txt` | Add `{nested_collection_candidates}` section |
| `graph_pipeline/prompts/schema_proposal_relationships.txt` | Rewrite implicit FK rule; expand path_fk_relationships to cover uniqueId-based nested FKs |
| `graph_pipeline/cypher_generator.py` | `generate_relationship_merge_batch` label-free MATCH (needed for empty to_type from Phase 3) |
