from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from kgent.agent.types import ModelBackend
from graph_pipeline.context_store import DatasetContext, HierarchyConfig, PathFKRelationship, SharedContext
from graph_pipeline.models import ExtractionSource, Node, Relationship

logger = logging.getLogger(__name__)

_ENTITY_EXTRACTION_BATCH_PROMPT = Path(__file__).parent / "prompts" / "entity_extraction_batch.txt"


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
# Rule 7: LLM-assisted extraction for ambiguous fields
# ---------------------------------------------------------------------------

async def _llm_extract_batch(
    batch: list[dict],
    dataset_ctx: DatasetContext,
    type_map: dict[str, str],
    backend: ModelBackend,
) -> tuple[list[Node], list[Relationship]]:
    """Send one batch of records to the LLM; parse the array response."""
    template = _ENTITY_EXTRACTION_BATCH_PROMPT.read_text(encoding="utf-8")
    dataset_id = dataset_ctx.dataset_id
    id_field = dataset_ctx.id_field
    ambiguous = dataset_ctx.ambiguous_fields

    allowed_node_labels = {nt.maps_to for nt in dataset_ctx.node_types}
    for nc in dataset_ctx.nested_collections:
        allowed_node_labels.add(nc.child_label)

    allowed_rel_types = {rt.maps_to for rt in dataset_ctx.relationship_types}
    for ir in dataset_ctx.implicit_relationships:
        allowed_rel_types.add(ir.maps_to)
    for pfk in dataset_ctx.path_fk_relationships:
        allowed_rel_types.add(pfk.maps_to)
    for nc in dataset_ctx.nested_collections:
        allowed_rel_types.add(nc.edge_type)
    if dataset_ctx.hierarchy_config:
        allowed_rel_types.add(dataset_ctx.hierarchy_config.edge_type)

    payload = [
        {
            "label": type_map.get(r.get(dataset_ctx.type_field, ""), r.get(dataset_ctx.type_field, "")),
            "record": r,
        }
        for r in batch
    ]
    prompt = template.format(
        dataset_id=dataset_id,
        id_field=id_field,
        ambiguous_fields=", ".join(ambiguous),
        known_node_labels_json=json.dumps(sorted(allowed_node_labels), ensure_ascii=False),
        known_rel_types_json=json.dumps(sorted(allowed_rel_types), ensure_ascii=False),
        records_json=json.dumps(payload, indent=2, ensure_ascii=False),
    )

    nodes: list[Node] = []
    rels: list[Relationship] = []
    try:
        response = await backend.complete(messages=[{"role": "user", "content": prompt}], tools=[])
        raw = response.content or ""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(text)
    except Exception as exc:
        logger.warning("LLM batch extraction failed: %s", exc)
        return nodes, rels

    if not isinstance(data, list):
        logger.warning("LLM batch extraction returned non-list: %r", type(data).__name__)
        return nodes, rels

    for item in data:
        if not isinstance(item, dict):
            continue
        record_id = item.get("source_record_id", "")
        for n in item.get("nodes", []):
            try:
                nodes.append(
                    Node(
                        id=n["id"],
                        label=n["label"],
                        properties=n.get("properties", {}),
                        source_record_id=n.get("source_record_id", record_id),
                        extraction_source=ExtractionSource.LLM_INFERRED,
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed node in batch result: %s", exc)
        for r_item in item.get("relationships", []):
            try:
                rels.append(
                    Relationship(
                        from_id=r_item["from_id"],
                        to_id=r_item["to_id"],
                        from_label=r_item["from_label"],
                        to_label=r_item["to_label"],
                        type=r_item["type"],
                        properties=r_item.get("properties", {}),
                        source_record_id=r_item.get("source_record_id", record_id),
                        extraction_source=ExtractionSource.LLM_INFERRED,
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed relationship in batch result: %s", exc)

    before_nodes, before_rels = len(nodes), len(rels)
    nodes = [n for n in nodes if n.label in allowed_node_labels]
    rels = [r for r in rels if r.type in allowed_rel_types]
    dropped = (before_nodes - len(nodes)) + (before_rels - len(rels))
    if dropped:
        logger.debug("Rule 7: filtered %d items with unknown labels/types", dropped)

    return nodes, rels


async def _llm_extract_ambiguous(
    records: list[dict],
    dataset_ctx: DatasetContext,
    type_map: dict[str, str],
    backend: ModelBackend,
    batch_size: int = 10,
    max_concurrency: int = 20,
) -> tuple[list[Node], list[Relationship]]:
    """Send ambiguous records to the LLM in batches; run all batches concurrently."""
    ambiguous = dataset_ctx.ambiguous_fields
    eligible = [r for r in records if any(f in r for f in ambiguous)]
    if not eligible:
        return [], []

    batches = [eligible[i : i + batch_size] for i in range(0, len(eligible), batch_size)]
    sem = asyncio.Semaphore(max_concurrency)

    async def _guarded(batch):
        async with sem:
            return await _llm_extract_batch(batch, dataset_ctx, type_map, backend)

    results = await asyncio.gather(
        *[_guarded(b) for b in batches],
        return_exceptions=True,
    )

    llm_nodes: list[Node] = []
    llm_rels: list[Relationship] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.warning("LLM batch raised: %s", result)
            continue
        nodes, rels = result
        llm_nodes.extend(nodes)
        llm_rels.extend(rels)

    return llm_nodes, llm_rels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_all(
    records: list[dict],
    dataset_ctx: DatasetContext,
    shared_ctx: SharedContext | None,
    backend: ModelBackend | None = None,
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

    # ----- Rule 7: LLM-assisted extraction for ambiguous fields --------------
    if dataset_ctx.ambiguous_fields and backend is not None:
        llm_nodes, llm_rels = await _llm_extract_ambiguous(
            records, dataset_ctx, type_map, backend
        )
        all_nodes.extend(llm_nodes)
        all_rels.extend(llm_rels)

    return all_nodes, all_rels
