from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterator

_TYPE_FIELD_CANDIDATES = ["typeName", "type", "kind", "__type", "category"]


def _detect_type_field(records: list[dict]) -> str | None:
    """Return the first candidate type field found in any record, or None."""
    for candidate in _TYPE_FIELD_CANDIDATES:
        if any(candidate in r for r in records):
            return candidate
    return None


def _truncate_nested_arrays(record: dict) -> dict:
    """Return a shallow copy of record with nested arrays-of-objects capped at 3 items."""
    result = {}
    for key, value in record.items():
        if (
            isinstance(value, list)
            and len(value) > 3
            and any(isinstance(item, dict) for item in value)
        ):
            result[key] = value[:3]
        else:
            result[key] = value
    return result


def sample_records(records: list[dict], n: int = 50, type_field: str | None = None) -> list[dict]:
    """Return up to n records, stratified by the type field when present.

    type_field: use this field for stratification. If None, auto-detect from common names.
    Guarantees at least 3 records per type (or all records for that type when fewer
    than 3 exist). Nested arrays of objects are truncated to 3 items per record.
    Falls back to simple random sampling when no type field is detected.
    """
    if not records:
        return []

    resolved_field = type_field or _detect_type_field(records)

    if resolved_field is None or not any(resolved_field in r for r in records):
        chosen = random.sample(records, min(n, len(records)))
        return [_truncate_nested_arrays(r) for r in chosen]

    # Group by type field — store indices into records, not object references
    by_type_indices: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_type_indices[r.get(resolved_field, "__untyped__")].append(i)

    # First pass: guarantee minimum 3 per type
    guaranteed_indices: dict[str, list[int]] = {}
    for type_name, indices in by_type_indices.items():
        guaranteed_indices[type_name] = random.sample(indices, min(3, len(indices)))

    guaranteed_count = sum(len(v) for v in guaranteed_indices.values())
    remaining_budget = max(0, n - guaranteed_count)

    # Second pass: proportional fill from the remainder
    pool_indices: list[int] = []
    for type_name, indices in by_type_indices.items():
        picked = set(guaranteed_indices[type_name])
        pool_indices.extend(i for i in indices if i not in picked)

    if pool_indices and remaining_budget > 0:
        extras = random.sample(pool_indices, min(remaining_budget, len(pool_indices)))
    else:
        extras = []

    all_indices: list[int] = [i for idxs in guaranteed_indices.values() for i in idxs]
    all_indices.extend(extras)
    return [_truncate_nested_arrays(records[i]) for i in all_indices]


def summarize_structure(
    records: list[dict],
    type_field: str | None = None,
    id_field: str | None = None,
) -> str:
    """Produce a compact structural summary suitable for injection into LLM prompts."""
    if not records:
        return "No records."

    resolved_type_field = type_field or _detect_type_field(records)

    # Collect all top-level keys
    all_keys: set[str] = set()
    for r in records:
        all_keys.update(r.keys())

    # Type distribution
    if resolved_type_field and any(resolved_type_field in r for r in records):
        type_counts: Counter = Counter(
            r.get(resolved_type_field) for r in records if resolved_type_field in r
        )
        type_field_label = resolved_type_field
    else:
        type_counts = Counter()
        type_field_label = None

    # FK candidate detection
    # Pass 1: suffix heuristic (Id / UniqueId)
    suffix_fk_keys = sorted(
        k for k in all_keys
        if (k.endswith("Id") or k.endswith("UniqueId"))
        and k != id_field
    )

    # Pass 2: sparse string fields (values are strings with no whitespace, length > 6,
    # appearing in fewer than 20% of records)
    total = len(records)
    threshold = max(1, int(total * 0.20))
    inferred_fk_keys: list[str] = []
    for key in sorted(all_keys):
        if key in suffix_fk_keys or key == id_field:
            continue
        values = [r[key] for r in records if key in r and isinstance(r[key], str)]
        if not values:
            continue
        sparse_strings = [v for v in values if len(v) > 6 and " " not in v]
        if sparse_strings and len(values) < threshold:
            inferred_fk_keys.append(key)

    # Fields containing nested arrays of objects
    nested_array_keys: set[str] = set()
    for r in records:
        for key, value in r.items():
            if isinstance(value, list) and any(isinstance(item, dict) for item in value):
                nested_array_keys.add(key)

    lines = []
    lines.append(f"Top-level keys ({len(all_keys)}): {', '.join(sorted(all_keys))}")

    if type_counts:
        dist = ", ".join(f"{t}({c})" for t, c in type_counts.most_common())
        lines.append(f"{type_field_label} distribution: {dist}")
    else:
        lines.append("type distribution: (no type field detected)")

    fk_parts = suffix_fk_keys + [f"{k} (inferred)" for k in inferred_fk_keys]
    if fk_parts:
        lines.append(f"Candidate FK fields: {', '.join(fk_parts)}")
    else:
        lines.append("Candidate FK fields: (none detected)")

    if nested_array_keys:
        lines.append(f"Nested array-of-object fields: {', '.join(sorted(nested_array_keys))}")
    else:
        lines.append("Nested array-of-object fields: (none)")

    return "\n".join(lines)


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


def compute_fingerprint(records: list[dict], type_field: str | None = None) -> str:
    """Return a 16-hex-char fingerprint of the type distribution of records.

    Uses the full records list (not a sample) so the fingerprint is deterministic
    across re-runs on the same file. Falls back to total record count when no type
    field is detected. Used by the ingest pipeline to skip schema discovery when
    the dataset is unchanged.
    """
    resolved = type_field or _detect_type_field(records)
    if resolved and any(resolved in r for r in records):
        counts: Counter = Counter(r.get(resolved, "__untyped__") for r in records)
    else:
        counts = Counter({"__total__": len(records)})
    all_keys = sorted({k for r in records for k in r.keys()})
    payload = json.dumps({"types": dict(counts), "keys": all_keys}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Streaming pre-scan (Pass 1)
# ---------------------------------------------------------------------------

@dataclass
class PrescanResult:
    sample: list[dict]
    fingerprint: str
    current_hashes: dict[str, str]
    ingest_ids: set[str]
    deleted_ids: set[str]
    type_field: str | None
    type_counts: Counter
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

        # Detect type field on the first record that has one
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

    # Fingerprint from type distribution + sorted key set of sample
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
