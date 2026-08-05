from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

from graph_pipeline.context_store import AmbiguousFieldRule, DatasetContext, HierarchyConfig, PathFKRelationship, SharedContext
from graph_pipeline.models import ExtractionSource, Node, Relationship
from graph_pipeline.neo4j_writer import WriteBuffer, WriteResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_nested(record: dict, dot_path: str):
    """Resolve a dot-separated path into a nested dict, e.g. 'a.b.c' → record['a']['b']['c'].
    Returns None if any segment is missing or not a dict."""
    parts = dot_path.split(".")
    current = record
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _scalar_properties(record: dict) -> dict:
    """Return only scalar (non-dict, non-list) fields."""
    return {
        k: v
        for k, v in record.items()
        if not isinstance(v, (dict, list))
    }


def _resolve_property_paths(record: dict, paths: list[str]) -> dict:
    """Resolve dot-path entries and return merged scalar properties.
    For each path:
    - If the resolved value is a dict, merge its scalar key-value pairs.
    - If the resolved value is a scalar (not dict/list), add it under the
      last path segment as key.
    Callers should apply top-level scalars after this result so that
    top-level values win on collision.
    """
    extra: dict = {}
    for path in paths:
        value = _get_nested(record, path)
        if value is None:
            continue
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(v, (dict, list)):
                    extra[k] = v
        elif not isinstance(value, list):
            extra[path.split(".")[-1]] = value
    return extra


def _node_type_map(dataset_ctx: DatasetContext) -> dict[str, str]:
    return {nt.name: nt.maps_to for nt in dataset_ctx.node_types}


def _rel_type_map(dataset_ctx: DatasetContext) -> dict[str, str]:
    return {rt.name: rt.maps_to for rt in dataset_ctx.relationship_types}


def _rel_label_map(dataset_ctx: DatasetContext) -> dict[str, tuple[str, str]]:
    return {rt.name: (rt.from_type, rt.to_type) for rt in dataset_ctx.relationship_types}


# ---------------------------------------------------------------------------
# Rules 3 + 4 helpers: path field → phantom nodes + hierarchy edges
# ---------------------------------------------------------------------------

def _build_hierarchy_structures(
    records: list[dict],
    config: HierarchyConfig,
    id_field: str,
    dataset_id: str,
    explicit_nodes_by_name: dict[str, Node],
) -> tuple[dict[str, Node], list[Relationship]]:
    """
    Parse all path values across records using config.field and config.separator.

    Returns:
        phantom_nodes: id → Node for each phantom node created
        hierarchy_rels: all hierarchy relationships derived from paths
    """
    phantom_nodes: dict[str, Node] = {}
    hierarchy_rels: list[Relationship] = []
    explicit_nodes_by_id: dict[str, Node] = {n.id: n for n in explicit_nodes_by_name.values()}

    def _resolve_segment(segment: str, record: dict | None = None) -> tuple[str, str]:
        """Return (node_id, node_label) for a path segment."""
        if record is not None:
            return f"{dataset_id}:{record.get(id_field, '')}", "leaf"

        explicit = explicit_nodes_by_name.get(segment)
        if explicit:
            return explicit.id, explicit.label

        phantom_id = f"{dataset_id}:path:{segment}"
        if phantom_id not in phantom_nodes:
            phantom_nodes[phantom_id] = Node(
                id=phantom_id,
                label=config.phantom_label,
                properties={"name": segment},
                source_record_id="",
                extraction_source=ExtractionSource.PHANTOM,
            )
        return phantom_id, config.phantom_label

    for record in records:
        path = record.get(config.field, "")
        if not path:
            continue
        segments = [s.strip() for s in path.split(config.separator) if s.strip()]
        if len(segments) < 2:
            continue

        for i in range(len(segments) - 1):
            parent_seg = segments[i]
            child_seg = segments[i + 1]
            is_leaf = (i + 1 == len(segments) - 1)

            parent_id, parent_label = _resolve_segment(parent_seg)
            if is_leaf:
                child_id = f"{dataset_id}:{record.get(id_field, '')}"
                leaf_explicit = explicit_nodes_by_id.get(child_id)
                child_label = leaf_explicit.label if leaf_explicit else config.phantom_label
            else:
                child_id, child_label = _resolve_segment(child_seg)

            hierarchy_rels.append(
                Relationship(
                    from_id=parent_id,
                    to_id=child_id,
                    from_label=parent_label,
                    to_label=child_label,
                    type=config.edge_type,
                    properties={},
                    source_record_id=record.get(id_field, ""),
                    extraction_source=ExtractionSource.RULE_BASED,
                )
            )

    return phantom_nodes, hierarchy_rels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_all(
    records: list[dict],
    dataset_ctx: DatasetContext,
    shared_ctx: SharedContext | None,
) -> tuple[list[Node], list[Relationship]]:
    """Apply all extraction rules in order. Returns (nodes, relationships)."""
    dataset_id = dataset_ctx.dataset_id
    id_field = dataset_ctx.id_field
    type_field = dataset_ctx.type_field
    type_map = _node_type_map(dataset_ctx)
    rel_map = _rel_type_map(dataset_ctx)
    rel_label_map = _rel_label_map(dataset_ctx)

    all_nodes: list[Node] = []
    all_rels: list[Relationship] = []

    # Pre-build path lookup indices for Rule 6b.
    # One index per distinct target_field used across path_fk_relationships.
    _path_indices: dict[str, dict[str, str]] = {}
    for pfk in dataset_ctx.path_fk_relationships:
        tf = pfk.target_field
        if tf not in _path_indices:
            _path_indices[tf] = {
                str(r[tf]): f"{dataset_id}:{r[id_field]}"
                for r in records
                if r.get(tf) is not None and r.get(id_field) is not None
            }

    # ----- Rule 1: id_field + type_field → Node --------------------------------
    for record in records:
        uid = record.get(id_field)
        type_name = record.get(type_field)
        if not uid or not type_name:
            continue
        label = type_map.get(type_name, type_name)
        all_nodes.append(
            Node(
                id=f"{dataset_id}:{uid}",
                label=label,
                properties={
                    **_resolve_property_paths(record, dataset_ctx.property_paths),
                    **_scalar_properties(record),
                },
                source_record_id=uid,
                extraction_source=ExtractionSource.RULE_BASED,
            )
        )

    # ----- Rule 2: nested_collections → child nodes + edges -------------------
    for record in records:
        parent_uid = record.get(id_field)
        parent_type = record.get(type_field)
        if not parent_uid:
            continue
        parent_label = type_map.get(parent_type, parent_type) if parent_type else ""

        for nc in dataset_ctx.nested_collections:
            items = _get_nested(record, nc.field)
            if not isinstance(items, list):
                continue
            for item in items:
                child_uid = item.get(nc.id_field)
                if not child_uid:
                    continue
                all_nodes.append(
                    Node(
                        id=f"{dataset_id}:{child_uid}",
                        label=nc.child_label,
                        properties={k: v for k, v in item.items() if not isinstance(v, (dict, list))},
                        source_record_id=parent_uid,
                        extraction_source=ExtractionSource.RULE_BASED,
                    )
                )
                all_rels.append(
                    Relationship(
                        from_id=f"{dataset_id}:{parent_uid}",
                        to_id=f"{dataset_id}:{child_uid}",
                        from_label=parent_label,
                        to_label=nc.child_label,
                        type=nc.edge_type,
                        properties={},
                        source_record_id=parent_uid,
                        extraction_source=ExtractionSource.RULE_BASED,
                    )
                )

    # ----- Rules 3 + 4: path hierarchy → phantom nodes + hierarchy edges ------
    if dataset_ctx.hierarchy_config is not None:
        explicit_by_name: dict[str, Node] = {}
        for node in all_nodes:
            name = node.properties.get("name")
            if name:
                explicit_by_name[name] = node

        phantom_nodes, hierarchy_rels = _build_hierarchy_structures(
            records, dataset_ctx.hierarchy_config, id_field, dataset_id, explicit_by_name
        )
        all_nodes.extend(phantom_nodes.values())
        all_rels.extend(hierarchy_rels)

    # ----- Rule 5: association array → explicit edges -------------------------
    ac = dataset_ctx.association_config
    if ac is not None:
        for record in records:
            this_uid = record.get(id_field)
            this_type = record.get(type_field)
            if not this_uid:
                continue
            this_label = type_map.get(this_type, this_type) if this_type else ""
            this_id = f"{dataset_id}:{this_uid}"

            for assoc in record.get(ac.array_field, []):
                edge_name = assoc.get(ac.edge_name_subfield, "")
                partner_id_raw = assoc.get(ac.partner_id_subfield)
                if ac.direction_subfield:
                    direction = assoc.get(ac.direction_subfield, ac.direction_default)
                else:
                    direction = ac.direction_default

                if partner_id_raw is None or str(partner_id_raw).strip() == "":
                    continue

                canonical_type = rel_map.get(edge_name)
                if not canonical_type:
                    continue

                from_label, to_label = rel_label_map.get(edge_name, ("", ""))
                partner_id = f"{dataset_id}:{partner_id_raw}"

                if direction == "out":
                    from_id, to_id = this_id, partner_id
                else:
                    from_id, to_id = partner_id, this_id

                all_rels.append(
                    Relationship(
                        from_id=from_id,
                        to_id=to_id,
                        from_label=from_label,
                        to_label=to_label,
                        type=canonical_type,
                        properties={},
                        source_record_id=this_uid,
                        extraction_source=ExtractionSource.RULE_BASED,
                    )
                )

    # ----- Rule 6: implicit foreign keys -------------------------------------
    for ir in dataset_ctx.implicit_relationships:
        fk_field = ir.edge_name
        rel_type = ir.maps_to
        target_ds = ir.target_dataset_id if ir.cross_dataset else dataset_id

        for record in records:
            this_uid = record.get(id_field)
            this_type = record.get(type_field)
            fk_value = record.get(fk_field)
            if not this_uid or fk_value is None:
                continue
            this_label = type_map.get(this_type, this_type) if this_type else ""

            all_rels.append(
                Relationship(
                    from_id=f"{dataset_id}:{this_uid}",
                    to_id=f"{target_ds}:{fk_value}",
                    from_label=this_label or ir.from_type,
                    to_label=ir.to_type,
                    type=rel_type,
                    properties={},
                    source_record_id=this_uid,
                    extraction_source=ExtractionSource.RULE_BASED,
                )
            )

    # ----- Rule 6b: path-valued FK relationships ----------------------------
    for pfk in dataset_ctx.path_fk_relationships:
        index = _path_indices.get(pfk.target_field, {})
        for record in records:
            this_uid = record.get(id_field)
            this_type = record.get(type_field)
            if not this_uid:
                continue
            this_label = type_map.get(this_type, this_type) if this_type else ""
            from_id = f"{dataset_id}:{this_uid}"
            if pfk.container_path is None:
                fk_value = record.get(pfk.fk_field)
                if not fk_value:
                    continue
                to_id = index.get(str(fk_value))
                if not to_id:
                    continue
                all_rels.append(
                    Relationship(
                        from_id=from_id,
                        to_id=to_id,
                        from_label=pfk.from_type or this_label,
                        to_label=pfk.to_type,
                        type=pfk.maps_to,
                        properties={},
                        source_record_id=this_uid,
                        extraction_source=ExtractionSource.RULE_BASED,
                    )
                )
            else:
                container = _get_nested(record, pfk.container_path)
                if not isinstance(container, list):
                    continue
                for item in container:
                    if not isinstance(item, dict):
                        continue
                    fk_value = item.get(pfk.fk_field)
                    if not fk_value:
                        continue
                    to_id = index.get(str(fk_value))
                    if not to_id:
                        continue
                    all_rels.append(
                        Relationship(
                            from_id=from_id,
                            to_id=to_id,
                            from_label=pfk.from_type or this_label,
                            to_label=pfk.to_type,
                            type=pfk.maps_to,
                            properties={},
                            source_record_id=this_uid,
                            extraction_source=ExtractionSource.RULE_BASED,
                        )
                    )

    # ----- Rule 7: deterministic ambiguous field relationships ---------------
    if dataset_ctx.ambiguous_field_rules:
        uid_set_local = {
            str(r.get(id_field, ""))
            for r in records if r.get(id_field) is not None
        }
        for record in records:
            r_uid = record.get(id_field)
            if not r_uid:
                continue
            r_type = record.get(type_field)
            r_label = type_map.get(r_type, r_type) if r_type else ""
            all_rels.extend(_apply_ambiguous_field_rules(
                record, dataset_id, str(r_uid), r_label,
                dataset_ctx.ambiguous_field_rules, uid_set_local,
            ))

    return all_nodes, all_rels


# ---------------------------------------------------------------------------
# Pass 2: index build for streaming extraction
# ---------------------------------------------------------------------------

@dataclass
class ExtractionIndices:
    # For Rule 3+4: {node_name: (namespaced_node_id, label)}
    name_to_node: dict[str, tuple[str, str]] = field(default_factory=dict)
    # For Rule 6b: {target_field: {field_value: namespaced_node_id}}
    path_value_index: dict[str, dict[str, str]] = field(default_factory=dict)
    # For Rule 7: raw (non-namespaced) UIDs for deterministic ambiguous field matching
    uid_set: set[str] = field(default_factory=set)


def build_extraction_indices(
    records_iter: Iterator[dict],
    dataset_ctx: DatasetContext,
) -> ExtractionIndices:
    """Pass 2: stream ingest records to build lookup indices for deferred rules.

    Runs in O(N) time and O(N) memory in index entries (not record size).
    Must complete before Pass 3 (extract_and_write_stream) begins.
    """
    indices = ExtractionIndices()
    dataset_id = dataset_ctx.dataset_id
    id_field = dataset_ctx.id_field
    type_field = dataset_ctx.type_field
    type_map = _node_type_map(dataset_ctx)

    # Initialise path_value_index keys from config so the key exists even when
    # no record has the target field.
    for pfk in dataset_ctx.path_fk_relationships:
        indices.path_value_index.setdefault(pfk.target_field, {})

    for record in records_iter:
        uid = record.get(id_field)
        type_name = record.get(type_field)
        if not uid or not type_name:
            continue

        label = type_map.get(type_name, type_name)
        namespaced_id = f"{dataset_id}:{uid}"

        indices.uid_set.add(str(uid))

        # name_to_node: used by hierarchy resolver (Rules 3+4)
        name = record.get("name")
        if name:
            indices.name_to_node[str(name)] = (namespaced_id, label)

        # path_value_index: used by Rule 6b
        for target_field in indices.path_value_index:
            val = record.get(target_field)
            if val is not None:
                indices.path_value_index[target_field][str(val)] = namespaced_id

    return indices


# ---------------------------------------------------------------------------
# Pass 3: streaming extraction + write
# ---------------------------------------------------------------------------

@dataclass
class StreamExtractResult:
    write_result: WriteResult
    # Deferred: (record_id, path_string, leaf_uid) — one per record with a path field
    path_tasks: list[tuple[str, str, str]] = field(default_factory=list)


async def _emit_hierarchy_inline(
    segments: list[str],
    leaf_uid: str,
    dataset_id: str,
    config: HierarchyConfig,
    name_to_node: dict[str, tuple[str, str]],
    buffer: WriteBuffer,
    phantom_nodes_seen: dict[str, bool],
    write_nodes: bool = True,
    write_rels: bool = True,
) -> None:
    """Emit hierarchy phantom nodes and CONTAINS edges inline.
    Node and rel writes are individually gated by write_nodes and write_rels."""
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
                if write_nodes:
                    await buffer.add_node(Node(
                        id=parent_id, label=config.phantom_label,
                        properties={"name": parent_seg}, source_record_id="",
                        extraction_source=ExtractionSource.PHANTOM,
                    ))

        # Resolve child
        if is_leaf:
            child_id = f"{dataset_id}:{leaf_uid}"
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
                if write_nodes:
                    await buffer.add_node(Node(
                        id=child_id, label=config.phantom_label,
                        properties={"name": child_seg}, source_record_id="",
                        extraction_source=ExtractionSource.PHANTOM,
                    ))

        if write_rels:
            await buffer.add_rel(Relationship(
                from_id=parent_id, to_id=child_id,
                from_label=parent_label, to_label=child_label,
                type=config.edge_type, properties={},
                source_record_id=leaf_uid,
                extraction_source=ExtractionSource.RULE_BASED,
            ))


def _apply_ambiguous_field_rules(
    record: dict,
    dataset_id: str,
    uid: str,
    this_label: str,
    rules: list[AmbiguousFieldRule],
    uid_set: set[str],
) -> list[Relationship]:
    rels: list[Relationship] = []
    for rule in rules:
        field_value = record.get(rule.field)
        if not isinstance(field_value, str) or not field_value:
            continue
        if rule.delimiter:
            tokens = [t.strip() for t in field_value.split(rule.delimiter) if t.strip()]
        else:
            tokens = [field_value.strip()] if field_value.strip() else []
        from_label = rule.from_type or this_label
        this_namespaced = f"{dataset_id}:{uid}"
        for token in tokens:
            if token not in uid_set:
                continue
            matched_id = f"{dataset_id}:{token}"
            if rule.direction == "out":
                from_id, to_id = this_namespaced, matched_id
            else:
                from_id, to_id = matched_id, this_namespaced
            rels.append(Relationship(
                from_id=from_id, to_id=to_id,
                from_label=from_label, to_label=rule.to_type,
                type=rule.rel_type, properties={},
                source_record_id=uid,
                extraction_source=ExtractionSource.RULE_BASED,
            ))
    return rels


async def extract_and_write_stream(
    records_iter: Iterator[dict],
    dataset_ctx: DatasetContext,
    shared_ctx: SharedContext | None,
    indices: ExtractionIndices,
    buffer: WriteBuffer,
    write_nodes: bool = True,
    write_rels: bool = True,
) -> StreamExtractResult:
    """Pass 3: stream ingest records, apply all rules inline. Writes are gated by
    write_nodes and write_rels to support two-pass (nodes-then-rels) orchestration.

    Rules 3+4 (hierarchy) are resolved inline when indices.name_to_node is
    populated (built in Pass 2). Otherwise path_tasks is populated for deferred
    post-stream resolution.
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
            if write_nodes:
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
                    if write_nodes:
                        await buffer.add_node(Node(
                            id=f"{dataset_id}:{child_uid}",
                            label=nc.child_label,
                            properties={k: v for k, v in item.items() if not isinstance(v, (dict, list))},
                            source_record_id=uid,
                            extraction_source=ExtractionSource.RULE_BASED,
                        ))
                    if write_rels:
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
                        await _emit_hierarchy_inline(
                            segments, uid, dataset_id, cfg, indices.name_to_node, buffer,
                            phantom_nodes_seen,
                            write_nodes=write_nodes,
                            write_rels=write_rels,
                        )
                    else:
                        result.path_tasks.append((uid, path, str(uid)))

        # Rule 5: associations
        ac = dataset_ctx.association_config
        if ac is not None and uid:
            this_label = type_map.get(type_name, type_name) if type_name else ""
            this_id = f"{dataset_id}:{uid}"
            for assoc in record.get(ac.array_field, []):
                edge_name = assoc.get(ac.edge_name_subfield, "")
                partner_id_raw = assoc.get(ac.partner_id_subfield)
                direction = (
                    assoc.get(ac.direction_subfield, ac.direction_default)
                    if ac.direction_subfield else ac.direction_default
                )
                if partner_id_raw is None or str(partner_id_raw).strip() == "":
                    continue
                canonical_type = rel_map.get(edge_name)
                if not canonical_type:
                    continue
                from_label, to_label = rel_label_map.get(edge_name, ("", ""))
                partner_id = f"{dataset_id}:{partner_id_raw}"
                from_id, to_id = (this_id, partner_id) if direction == "out" else (partner_id, this_id)
                if write_rels:
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
            if write_rels:
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
                        if write_rels:
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
                                    if write_rels:
                                        await buffer.add_rel(Relationship(
                                            from_id=from_id, to_id=to_id,
                                            from_label=pfk.from_type or this_label, to_label=pfk.to_type,
                                            type=pfk.maps_to, properties={},
                                            source_record_id=uid,
                                            extraction_source=ExtractionSource.RULE_BASED,
                                        ))

        # Rule 7: deterministic ambiguous field relationships
        if write_rels and dataset_ctx.ambiguous_field_rules and uid:
            this_label_r7 = type_map.get(type_name, type_name) if type_name else ""
            for _rel in _apply_ambiguous_field_rules(
                record, dataset_id, uid, this_label_r7,
                dataset_ctx.ambiguous_field_rules, indices.uid_set,
            ):
                await buffer.add_rel(_rel)

    await buffer.flush_all()

    return result
