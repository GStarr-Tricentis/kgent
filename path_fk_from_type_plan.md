# Plan: Deterministic path FK from_type validation

## Problem

`path_fk_relationships` have a `from_type` field that tells the extractor which label
to use for the source node of the relationship. When the LLM gets this wrong (e.g.
`from_type: "TestCase"` when the array actually lives on RTSB records), the Cypher
generator emits `MATCH (a:TestCase {id: "<rtsb-id>"})`, Neo4j finds nothing, and every
rel in that batch is silently dropped.

This is the same structural problem as heterogeneous `to_type` on association rels —
the LLM is guessing a fact that is directly readable from the sample. The fix is the
same pattern: a deterministic post-Call-2 validator that reads the sample and corrects
what the LLM got wrong.

## What the validator does

For each `PathFKRelationship` where `container_path` is set and `from_type` is
non-empty, scan the sample to find which `typeName` values actually have a non-empty
array at that `container_path`. If `from_type` doesn't appear among the observed types,
clear it to `""`. An empty `from_type` makes the extractor fall back to
`this_label` — the record's own type — which is always correct.

No prompt changes. No extractor changes. Pure post-call correction in schema_discovery.

---

## Phase 1 — Private helper `_resolve_dot_path`

`_get_nested` already exists in `extractor.py` but importing it would create a
cross-module dependency. Inline an equivalent private version in `schema_discovery.py`
immediately before `_scan_nested_collection_candidates`:

```python
def _resolve_dot_path(record: dict, path: str):
    """Resolve a dot-separated path into a nested structure.
    Returns None if any segment is missing or not a dict."""
    current = record
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
```

---

## Phase 2 — New helper `_validate_path_fk_from_types`

Add to `schema_discovery.py` immediately before `_validate_association_partner_types`:

```python
def _validate_path_fk_from_types(
    path_fk_rels: list[PathFKRelationship],
    sample: list[dict],
    type_field: str,
) -> list[PathFKRelationship]:
    """Clear from_type on path FKs where the LLM's guess doesn't match the sample.

    For each path FK with a non-empty container_path and from_type, scans the sample
    to find which record types actually have non-empty arrays at that path. If from_type
    doesn't appear among the observed types, clears it to "" so the extractor uses the
    record's own type (this_label) instead — which is always correct.
    """
    if not path_fk_rels:
        return path_fk_rels
    if not any(record.get(type_field) for record in sample):
        return path_fk_rels

    updated = []
    for pfk in path_fk_rels:
        if not pfk.container_path or not pfk.from_type:
            updated.append(pfk)
            continue

        observed_types: set[str] = set()
        for record in sample:
            type_name = record.get(type_field)
            if not type_name:
                continue
            value = _resolve_dot_path(record, pfk.container_path)
            if isinstance(value, list) and value:
                observed_types.add(str(type_name))

        if pfk.from_type not in observed_types:
            logger.warning(
                "path_fk '%s' from_type '%s' not observed in sample "
                "(types with non-empty '%s': %s) — clearing to use record's own type",
                pfk.maps_to, pfk.from_type, pfk.container_path, sorted(observed_types),
            )
            updated.append(pfk.model_copy(update={"from_type": ""}))
        else:
            updated.append(pfk)

    return updated
```

---

## Phase 3 — Wire into `propose_dataset_context`

Add immediately after the `_validate_association_partner_types` call:

```python
path_fk_rels = _validate_path_fk_from_types(
    path_fk_rels,
    sample,
    type_field=structural_config.get("type_field") or "typeName",
)
```

---

## Phase 4 — Tests

Add `TestValidatePathFkFromTypes` to `tests/test_schema_discovery.py` after
`TestValidateAssociationPartnerTypes`:

```python
class TestValidatePathFkFromTypes:
    def _make_pfk(self, container_path, from_type, maps_to="STEP_USES_MODULE"):
        from graph_pipeline.context_store import PathFKRelationship
        return PathFKRelationship(
            container_path=container_path,
            fk_field="moduleUniqueId",
            target_field="uniqueId",
            maps_to=maps_to,
            from_type=from_type,
            to_type="",
        )

    def test_correct_from_type_preserved(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "typeName": "RTSB",
             "details": {"testSteps": [{"uniqueId": "s1"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "RTSB")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == "RTSB"

    def test_wrong_from_type_cleared(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "typeName": "RTSB",
             "details": {"testSteps": [{"uniqueId": "s1"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "TestCase")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == ""

    def test_empty_from_type_unchanged(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "typeName": "RTSB",
             "details": {"testSteps": [{"uniqueId": "s1"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == ""

    def test_null_container_path_unchanged(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        pfk = self._make_pfk(None, "TestCase")
        result = _validate_path_fk_from_types([pfk], [{"uniqueId": "r1", "typeName": "TestCase"}], "typeName")
        assert result[0].from_type == "TestCase"

    def test_multiple_types_have_array_from_type_matches_one(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "typeName": "RTSB",
             "details": {"testSteps": [{"uniqueId": "s1"}]}},
            {"uniqueId": "r2", "typeName": "TestCase",
             "details": {"testSteps": [{"uniqueId": "s2"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "RTSB")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == "RTSB"

    def test_empty_array_not_counted(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "typeName": "TestCase",
             "details": {"testSteps": []}},          # empty — doesn't count
            {"uniqueId": "r2", "typeName": "RTSB",
             "details": {"testSteps": [{"uniqueId": "s1"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "TestCase")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == ""

    def test_empty_path_fk_rels_returns_empty(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        assert _validate_path_fk_from_types([], [], "typeName") == []

    def test_no_type_field_in_sample_returns_unchanged(self):
        from graph_pipeline.schema_discovery import _validate_path_fk_from_types
        sample = [
            {"uniqueId": "r1", "details": {"testSteps": [{"uniqueId": "s1"}]}},
        ]
        pfk = self._make_pfk("details.testSteps", "TestCase")
        result = _validate_path_fk_from_types([pfk], sample, "typeName")
        assert result[0].from_type == "TestCase"
```

---

## Files changed

| File | Change |
|---|---|
| `graph_pipeline/schema_discovery.py` | Add `_resolve_dot_path`; add `_validate_path_fk_from_types`; wire into `propose_dataset_context` |
| `tests/test_schema_discovery.py` | Add `TestValidatePathFkFromTypes` |

No prompt changes. No extractor changes. No context_store changes.
