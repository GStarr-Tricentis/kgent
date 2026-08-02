from __future__ import annotations

import contextlib
import datetime
import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DATASET_CONTEXT_SCHEMA_VERSION = 1
SHARED_CONTEXT_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SharedNodeType(BaseModel):
    name: str
    description: str = ""
    identity_key: str = "uniqueId"
    source_datasets: list[str] = Field(default_factory=list)
    maps_to: str = ""  # canonical label; populated when merging from dataset contexts


class SharedRelationshipType(BaseModel):
    name: str
    description: str = ""
    from_type: str = Field("", alias="from")
    to_type: str = Field("", alias="to")
    source_datasets: list[str] = Field(default_factory=list)
    maps_to: str = ""  # canonical label

    model_config = {"populate_by_name": True}


class StructuralPattern(BaseModel):
    name: str
    description: str = ""
    applies_to: str | list[str] = "all"


class SharedContext(BaseModel):
    version: int = 0
    updated_at: str = ""
    schema_version: int = 0
    node_types: list[SharedNodeType] = Field(default_factory=list)
    relationship_types: list[SharedRelationshipType] = Field(default_factory=list)
    structural_patterns: list[StructuralPattern] = Field(default_factory=list)


class DatasetNodeType(BaseModel):
    name: str
    maps_to: str
    identity_key: str = "uniqueId"


class DatasetRelationshipType(BaseModel):
    name: str
    maps_to: str
    from_type: str = Field("", alias="from")
    to_type: str = Field("", alias="to")

    model_config = {"populate_by_name": True}


class ImplicitRelationship(BaseModel):
    description: str
    pattern: str
    edge_name: str
    maps_to: str
    from_type: str = ""
    to_type: str = ""
    cross_dataset: bool = False
    target_dataset_id: str | None = None


class PathFKRelationship(BaseModel):
    container_path: str | None = None
    fk_field: str
    target_field: str = "nodePath"
    maps_to: str
    from_type: str = ""
    to_type: str = ""


class DesignDecision(BaseModel):
    question: str
    decision: str
    rationale: str = ""


class NestedCollection(BaseModel):
    field: str
    child_label: str
    edge_type: str
    id_field: str = "uniqueId"


class AssociationConfig(BaseModel):
    array_field: str = "associations"
    edge_name_subfield: str = "edgeName"
    partner_id_subfield: str = "partnerId"
    direction_subfield: str = "direction"
    direction_default: str = "out"


class HierarchyConfig(BaseModel):
    field: str
    separator: str = "/"
    phantom_label: str = "Folder"
    edge_type: str = "CONTAINS"


class DatasetContext(BaseModel):
    dataset_id: str
    source_file: str = ""
    generated_at: str = ""
    id_field: str = "uniqueId"
    type_field: str = "typeName"
    node_types: list[DatasetNodeType] = Field(default_factory=list)
    relationship_types: list[DatasetRelationshipType] = Field(default_factory=list)
    implicit_relationships: list[ImplicitRelationship] = Field(default_factory=list)
    path_fk_relationships: list[PathFKRelationship] = Field(default_factory=list)
    nested_collections: list[NestedCollection] = Field(default_factory=list)
    association_config: AssociationConfig | None = None
    hierarchy_config: HierarchyConfig | None = None
    design_decisions: list[DesignDecision] = Field(default_factory=list)
    ambiguous_fields: list[str] = Field(default_factory=list)
    property_paths: list[str] = Field(default_factory=list)
    source_fingerprint: str = ""
    schema_version: int = 0


# ---------------------------------------------------------------------------
# MergeConflict — raised (and is an Exception) so callers can catch it
# ---------------------------------------------------------------------------

@dataclass
class MergeConflict(Exception):
    type: Literal["node", "relationship"]
    source_name: str
    existing_canonical: str
    proposed_canonical: str
    existing_dataset: str
    new_dataset: str

    def __str__(self) -> str:
        return (
            f"MergeConflict({self.type!r}): '{self.source_name}' maps to "
            f"'{self.existing_canonical}' in {self.existing_dataset!r} but "
            f"'{self.proposed_canonical}' in {self.new_dataset!r}"
        )


# ---------------------------------------------------------------------------
# Path resolution — override via GRAPH_PIPELINE_CONTEXT_DIR for tests
# ---------------------------------------------------------------------------

def _context_dir() -> Path:
    env = os.environ.get("GRAPH_PIPELINE_CONTEXT_DIR")
    if env:
        return Path(env)
    return Path("context")


def _shared_path() -> Path:
    return _context_dir() / "shared_context.yaml"


@contextlib.contextmanager
def _shared_write_lock():
    """Exclusive advisory lock on shared_context.yaml for the duration of a write."""
    lock_path = _shared_path().with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        yield


def _dataset_path(dataset_id: str) -> Path:
    return _context_dir() / "datasets" / f"{dataset_id}.yaml"


def _hash_store_path(dataset_id: str) -> Path:
    return _context_dir() / "datasets" / f"{dataset_id}_hashes.json"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _shared_to_dict(sc: SharedContext) -> dict:
    """Serialize SharedContext to a plain dict for YAML round-tripping."""
    return {
        "version": sc.version,
        "updated_at": sc.updated_at,
        "schema_version": SHARED_CONTEXT_SCHEMA_VERSION,
        "node_types": [
            {
                "name": nt.name,
                "description": nt.description,
                "identity_key": nt.identity_key,
                "maps_to": nt.maps_to,
                "source_datasets": nt.source_datasets,
            }
            for nt in sc.node_types
        ],
        "relationship_types": [
            {
                "name": rt.name,
                "description": rt.description,
                "from": rt.from_type,
                "to": rt.to_type,
                "maps_to": rt.maps_to,
                "source_datasets": rt.source_datasets,
            }
            for rt in sc.relationship_types
        ],
        "structural_patterns": [
            {"name": p.name, "description": p.description, "applies_to": p.applies_to}
            for p in sc.structural_patterns
        ],
    }


def _shared_from_dict(data: dict) -> SharedContext:
    node_types = []
    for nt in data.get("node_types", []):
        node_types.append(SharedNodeType(**nt))

    rel_types = []
    for rt in data.get("relationship_types", []):
        rel_types.append(SharedRelationshipType(**rt))

    patterns = []
    for p in data.get("structural_patterns", []):
        patterns.append(StructuralPattern(**p))

    return SharedContext(
        version=data.get("version", 0),
        updated_at=data.get("updated_at", ""),
        schema_version=data.get("schema_version", 0),
        node_types=node_types,
        relationship_types=rel_types,
        structural_patterns=patterns,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_shared_context() -> SharedContext:
    path = _shared_path()
    if not path.exists():
        return SharedContext()
    try:
        sc = _shared_from_dict(_load_yaml(path))
    except Exception as exc:
        raise ValueError(
            f"Could not load shared context from {path}: {exc}"
        ) from exc
    if sc.schema_version > SHARED_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            f"Shared context at {path} was written by a newer version of kgent "
            f"(schema_version={sc.schema_version}, current={SHARED_CONTEXT_SCHEMA_VERSION}). "
            "Upgrade kgent or delete the file to regenerate."
        )
    return sc


def load_dataset_context(dataset_id: str) -> DatasetContext | None:
    path = _dataset_path(dataset_id)
    if not path.exists():
        return None
    data = _load_yaml(path)
    try:
        ctx = DatasetContext(**data)
    except Exception as exc:
        raise ValueError(
            f"Could not load dataset context from {path}: {exc}\n"
            "Fix the YAML file and try again, or delete it to regenerate from scratch."
        ) from exc
    if ctx.schema_version > DATASET_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            f"Dataset context at {path} was written by a newer version of kgent "
            f"(schema_version={ctx.schema_version}, current={DATASET_CONTEXT_SCHEMA_VERSION}). "
            "Upgrade kgent or delete the file to regenerate."
        )
    return ctx


def save_dataset_context(ctx: DatasetContext) -> None:
    path = _dataset_path(ctx.dataset_id)
    data = ctx.model_dump()
    data["schema_version"] = DATASET_CONTEXT_SCHEMA_VERSION
    _save_yaml(path, data)


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
    """Merge dataset_ctx into shared_context, enforcing canonical label consistency.

    The read-modify-write is protected by an exclusive file lock on
    shared_context.lock, preventing data loss when two ingest processes run
    concurrently. Raises MergeConflict if the same source name already maps to a
    different canonical label. Never writes to disk when a conflict is detected.
    """
    with _shared_write_lock():
        sc = load_shared_context()

        # Build lookup: source_name → (canonical, first_dataset_id)
        node_index: dict[str, tuple[str, str]] = {
            nt.name: (nt.maps_to, nt.source_datasets[0] if nt.source_datasets else "")
            for nt in sc.node_types
        }
        rel_index: dict[str, tuple[str, str]] = {
            rt.name: (rt.maps_to, rt.source_datasets[0] if rt.source_datasets else "")
            for rt in sc.relationship_types
        }

        # Validate all incoming types before mutating anything
        for nt in dataset_ctx.node_types:
            if nt.name in node_index:
                existing_canonical, existing_ds = node_index[nt.name]
                if existing_canonical != nt.maps_to:
                    raise MergeConflict(
                        type="node",
                        source_name=nt.name,
                        existing_canonical=existing_canonical,
                        proposed_canonical=nt.maps_to,
                        existing_dataset=existing_ds,
                        new_dataset=dataset_ctx.dataset_id,
                    )

        for rt in dataset_ctx.relationship_types:
            if rt.name in rel_index:
                existing_canonical, existing_ds = rel_index[rt.name]
                if existing_canonical != rt.maps_to:
                    raise MergeConflict(
                        type="relationship",
                        source_name=rt.name,
                        existing_canonical=existing_canonical,
                        proposed_canonical=rt.maps_to,
                        existing_dataset=existing_ds,
                        new_dataset=dataset_ctx.dataset_id,
                    )

        # No conflicts — apply mutations
        for nt in dataset_ctx.node_types:
            existing = next((x for x in sc.node_types if x.name == nt.name), None)
            if existing is None:
                sc.node_types.append(
                    SharedNodeType(
                        name=nt.name,
                        identity_key=nt.identity_key,
                        maps_to=nt.maps_to,
                        source_datasets=[dataset_ctx.dataset_id],
                    )
                )
            else:
                if dataset_ctx.dataset_id not in existing.source_datasets:
                    existing.source_datasets.append(dataset_ctx.dataset_id)

        for rt in dataset_ctx.relationship_types:
            existing = next((x for x in sc.relationship_types if x.name == rt.name), None)
            if existing is None:
                sc.relationship_types.append(
                    SharedRelationshipType(
                        name=rt.name,
                        maps_to=rt.maps_to,
                        **{"from": rt.from_type, "to": rt.to_type},
                        source_datasets=[dataset_ctx.dataset_id],
                    )
                )
            else:
                if dataset_ctx.dataset_id not in existing.source_datasets:
                    existing.source_datasets.append(dataset_ctx.dataset_id)

        sc.version += 1
        sc.updated_at = str(datetime.date.today())

        _save_yaml(_shared_path(), _shared_to_dict(sc))
        return sc
